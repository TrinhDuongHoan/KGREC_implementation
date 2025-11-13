import os
import sys
import csv
import random
from time import time
import time as ttime
import psutil
import yaml
import argparse
import logging
import pathlib

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm
import scipy.sparse as sp

# --- sys.path ---
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loaders.kgat_loader import DataLoaderKGAT
from models.KGAT import KGAT

from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *


def _torch_sparse_to_scipy(A_sparse_torch):

    A_sparse_torch = A_sparse_torch.coalesce().cpu()
    idx = A_sparse_torch.indices().numpy()
    val = A_sparse_torch.values().numpy()
    shape = tuple(A_sparse_torch.shape)
    return sp.coo_matrix((val, (idx[0], idx[1])), shape=shape)


@torch.no_grad()
def _kgat_compute_embeddings(model, alg_type):
    """Tính UA, EA embeddings một lần để dùng cho predict toàn bộ items."""
    if alg_type in ['bi', 'kgat']:
        ua, ea = model._create_bi_interaction_embed()
    elif alg_type in ['gcn']:
        ua, ea = model._create_gcn_embed()
    elif alg_type in ['graphsage']:
        ua, ea = model._create_graphsage_embed()
    else:
        raise ValueError("alg_type invalid.")
    return ua, ea


def evaluate(model, dataloader, Ks, device):
    """Full ranking: u_emb @ i_emb^T. Không gom điểm sang numpy để tiết kiệm RAM."""
    eval_t0 = ttime.perf_counter()
    model._training_flag = False
    model_device = device

    # tính embed một lần
    ua, ea = _kgat_compute_embeddings(model, model.alg_type)
    ua = ua.to(model_device)
    ea = ea.to(model_device)

    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    user_ids = list(test_user_dict.keys())
    user_batches = [user_ids[i:i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]

    n_items = dataloader.n_items
    # item ids được giả định là [0..n_items-1] trong entity space
    all_item_ids = torch.arange(n_items, dtype=torch.long, device=model_device)

    metric_names = ['precision', 'recall', 'f1', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    with tqdm(total=len(user_batches), desc='Evaluating Iteration') as pbar:
        for batch_u in user_batches:
            bu = torch.as_tensor(batch_u, dtype=torch.long, device=model_device)
            # Map KG-space user ids back to [0, n_users) for UA indexing
            if bu.numel() > 0 and int(bu.min().item()) >= dataloader.n_entities:
                bu_idx = bu - dataloader.n_entities
            else:
                # mixed or already user-space
                bu_idx = torch.clamp(bu, min=0, max=dataloader.n_users - 1)

            # scores: [B, I]
            scores = ua[bu_idx] @ ea[all_item_ids].T  # [B, D] @ [I, D]^T
            # mask train items
            for row, u in enumerate(batch_u):
                tr = train_user_dict[u]
                if len(tr) > 0:
                    scores[row, torch.as_tensor(tr, device=model_device, dtype=torch.long)] = -float("inf")

            # metrics
            batch_metrics = calc_metrics_at_k(
                scores, train_user_dict, test_user_dict,
                np.asarray(batch_u, dtype=np.int64),
                np.arange(n_items, dtype=np.int64),
                Ks
            )

            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])

            pbar.update(1)

    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    eval_time_s = max(ttime.perf_counter() - eval_t0, 1e-9)
    return metrics_dict, eval_time_s


def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Load data -----
    data = DataLoaderKGAT(args, logging)

    # ----- Pretrain embeddings -----
    pretrain_data = None
    if args.use_pretrain == 1:
        pretrain_data = {
            'user_embed': data.user_pre_embed.astype(np.float32),
            'item_embed': data.item_pre_embed.astype(np.float32)
        }

    # ----- Build data_config cho KGAT -----
    # A_in từ loader là torch.sparse; KGAT cần scipy.sparse
    A_coo = _torch_sparse_to_scipy(data.A_in)

    # list cạnh để update attention
    # Dùng các triple đã “train”: h_list, r_list, t_list
    all_h_list = data.h_list.cpu().numpy().astype(np.int64)
    all_t_list = data.t_list.cpu().numpy().astype(np.int64)
    all_r_list = data.r_list.cpu().numpy().astype(np.int64)

    # A values ban đầu: nếu muốn bám theo A_coo, map theo (h,t)
    # Ở đây khởi tạo ones theo paper (ổn định, không phụ thuộc chuẩn hoá trước).
    all_v_list = np.ones_like(all_h_list, dtype=np.float32)

    data_config = {
        'n_users': data.n_users,
        'n_items': data.n_items,
        'n_entities': data.n_entities,
        'n_relations': data.n_relations,
        'A_in': A_coo,
        'all_h_list': all_h_list,
        'all_r_list': all_r_list,
        'all_t_list': all_t_list,
        'all_v_list': all_v_list,
    }

    # ----- Init model -----
    model = KGAT(data_config, pretrain_data, args)
    model.device = device
    # các weights đã nằm trong model.weights (ParameterDict) và optimizer bên trong
    logging.info("KGAT initialized.")

    # --- model stats: number of trainable params and approximate model size (MB) ---
    model_n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    model_size_mb = model_param_bytes / (1024.0 * 1024.0)
    logging.info(f"MODEL STATS | trainable_params: {model_n_params:,} | approx_size: {model_size_mb:.2f} MB")

    best_epoch = -1
    best_recall = 0.0

    Ks = eval(args.Ks) if isinstance(args.Ks, str) else list(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    epoch_list = []
    metrics_list = {k: {'precision': [], 'recall': [], 'f1': [], 'ndcg': []} for k in Ks}
    training_loss = {'epoch': [], 'cf_loss': [], 'kg_loss': [], 'total_loss': []}

    os.makedirs(args.save_dir, exist_ok=True)
    # Overall runtime trackers
    overall_t0 = ttime.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_gpu_peak_bytes = 0
    eval_time_acc = 0.0
    last_eval_time_s = 0.0

    # ====== Loop ======
    for epoch in range(1, args.n_epoch + 1):
        time0 = time()

        # ---------- Phase I: CF ----------
        time1 = time()
        cf_total_loss = 0.0
        n_cf_batch = data.n_cf_train // data.batch_size + 1

        for it in range(1, n_cf_batch + 1):
            t_batch_wall = ttime.perf_counter()

            users, pos_i, neg_i = data.generate_cf_batch(data.train_user_dict, data.batch_size)
            feed = {
                model.users: users.numpy(),
                model.pos_items: pos_i.numpy(),
                model.neg_items: neg_i.numpy(),
                model.node_dropout: np.zeros(model.n_layers, dtype=np.float32),
                model.mess_dropout: np.zeros(model.n_layers, dtype=np.float32),
            }

            t_step = ttime.perf_counter()
            _, loss, base, kge, reg = model.train(None, feed)
            if np.isnan(loss):
                logging.info(f'ERROR (CF Training): Epoch {epoch:04d} Iter {it:04d}/{n_cf_batch:04d} Loss is nan.')
                sys.exit(1)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            cf_total_loss += loss
            if (it % args.cf_print_every) == 0:
                logging.info(f'CF Training: Epoch {epoch:04d} Iter {it:04d}/{n_cf_batch:04d} | '
                             f'Iter Loss {loss:.4f} | Mean {cf_total_loss/it:.4f}')

        logging.info(f'CF Training: Epoch {epoch:04d} Total Iter {n_cf_batch:04d} | '
                     f'Total Time {time() - time1:.1f}s | Mean Loss {cf_total_loss / max(1, n_cf_batch):.4f}')

        # ---------- Phase II: KG ----------
        time3 = time()
        kg_total_loss = 0.0
        n_kg_batch = data.n_kg_train // args.batch_size_kg + 1

        for it in range(1, n_kg_batch + 1):
            t_batch_wall = ttime.perf_counter()

            h, r, pos_t, neg_t = data.generate_kg_batch(data.train_kg_dict, args.batch_size_kg, data.n_users_entities)
            feed = {
                model.h: h.numpy(),
                model.r: r.numpy(),
                model.pos_t: pos_t.numpy(),
                model.neg_t: neg_t.numpy(),
            }

            t_step = ttime.perf_counter()
            _, loss2, kge2, reg2 = model.train_A(None, feed)
            if np.isnan(loss2):
                logging.info(f'ERROR (KG Training): Epoch {epoch:04d} Iter {it:04d}/{n_kg_batch:04d} Loss is nan.')
                sys.exit(1)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            kg_total_loss += loss2
            if (it % args.kg_print_every) == 0:
                logging.info(f'KG Training: Epoch {epoch:04d} Iter {it:04d}/{n_kg_batch:04d} | '
                             f'Iter Loss {loss2:.4f} | Mean {kg_total_loss/it:.4f}')

        logging.info(f'KG Training: Epoch {epoch:04d} Total Iter {n_kg_batch:04d} | '
                     f'Total Time {time() - time3:.1f}s | Mean Loss {kg_total_loss / max(1, n_kg_batch):.4f}')

        # snapshot training peak so far
        if torch.cuda.is_available():
            train_gpu_peak_bytes = max(train_gpu_peak_bytes, torch.cuda.max_memory_allocated())

        training_loss['epoch'].append(epoch)
        training_loss['cf_loss'].append(cf_total_loss / max(1, n_cf_batch))
        training_loss['kg_loss'].append(kg_total_loss / max(1, n_kg_batch))
        training_loss['total_loss'].append(training_loss['cf_loss'][-1] + training_loss['kg_loss'][-1])

        # ---------- Update attentive A ----------
        t5 = time()
        model.update_attentive_A(None)
        logging.info('Update Attention done: Epoch {:04d} | Time {:.1f}s'.format(epoch, time() - t5))

        # ---------- Evaluate ----------
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()  # avoid polluting training peak with eval
            t6 = ttime.perf_counter()
            metrics, eval_time_s = evaluate(model, data, Ks, device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            last_eval_time_s = max(eval_time_s, 1e-9)
            eval_time_acc += last_eval_time_s
            logging.info('CF Evaluation: Epoch {:04d} | Time {:.1f}s | '
                         'Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], F1 [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                epoch, last_eval_time_s,
                metrics[k_min]['precision'], metrics[k_max]['precision'],
                metrics[k_min]['recall'], metrics[k_max]['recall'],
                metrics[k_min]['f1'], metrics[k_max]['f1'],
                metrics[k_min]['ndcg'], metrics[k_max]['ndcg'])
            )

            epoch_list.append(epoch)
            for k in Ks:
                for m in ['precision', 'recall', 'f1', 'ndcg']:
                    metrics_list[k][m].append(metrics[k][m])

            best_recall, should_stop = early_stopping(metrics_list[k_min]['recall'], args.stopping_steps)
            if should_stop:
                break

            if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:
                save_model(model, args.save_dir, epoch, -1 if 'best_epoch' not in locals() else best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

    # ----- Save loss -----
    training_loss_df = pd.DataFrame(training_loss)
    training_loss_df.to_csv(os.path.join(args.save_dir, "training_loss.csv"), index=False)

    # ----- Save metrics history -----
    metrics_records = []
    for i, ep in enumerate(epoch_list):
        row = {"epoch_idx": ep}
        for k in Ks:
            for m in ["precision", "recall", "f1", "ndcg"]:
                if i < len(metrics_list[k][m]):
                    row[f"{m}@{k}"] = metrics_list[k][m][i]
        metrics_records.append(row)

    metrics_df = pd.DataFrame(metrics_records)
    csv_path = os.path.join(args.save_dir, "kgat_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)

    # ----- Best line -----
    if len(metrics_df) > 0 and 'best_epoch' in locals() and best_epoch in metrics_df["epoch_idx"].values:
        best_metrics = metrics_df.loc[metrics_df["epoch_idx"] == best_epoch].iloc[0].to_dict()
        logging.info(
            "Best CF Evaluation: Epoch {:04d} | "
            "Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], F1 [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]".format(
                int(best_metrics["epoch_idx"]),
                best_metrics[f"precision@{k_min}"], best_metrics[f"precision@{k_max}"],
                best_metrics[f"recall@{k_min}"], best_metrics[f"recall@{k_max}"],
                best_metrics[f"f1@{k_min}"], best_metrics[f"f1@{k_max}"],
                best_metrics[f"ndcg@{k_min}"], best_metrics[f"ndcg@{k_max}"]
            )
        )
    else:
        logging.info("No best epoch recorded; metrics history may be empty.")

    # Overall runtime summary (one line CSV)
    total_wall_s = ttime.perf_counter() - overall_t0
    total_train_s = max(total_wall_s - eval_time_acc, 0.0)
    train_gpu_peak_mb = (train_gpu_peak_bytes / 1024 / 1024) if torch.cuda.is_available() else 0.0
    overall_csv = os.path.join(args.save_dir, "runtime_overall.csv")
    with open(overall_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["total_train_s", "peak_train_gpu_MB", "inference_time_s", "n_params", "model_size_MB"]) 
        writer.writerow([
            round(total_train_s, 6),
            round(train_gpu_peak_mb, 2),
            round(last_eval_time_s, 6),
            int(model_n_params),
            round(model_size_mb, 2)
        ])
    logging.info("RUNTIME SUMMARY | train_total_s: {:.2f} | peak_train_gpu_MB: {:.1f} | inference_time_s: {:.2f} | params: {} | model_size_MB: {:.2f}".format(
        total_train_s, train_gpu_peak_mb, last_eval_time_s, model_n_params, model_size_mb
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, required=True, help="Path to YAML config file")
    cli_args = parser.parse_args()

    with open(cli_args.configs, 'r') as f:
        cfg = yaml.safe_load(f)
    # Normalize/alias config fields for compatibility
    # KGAT model expects embed_size; some configs may provide embed_dim
    if 'embed_size' not in cfg and 'embed_dim' in cfg:
        cfg['embed_size'] = cfg['embed_dim']
    # KGAT model expects kge_size; some configs may provide kge_dim or relation_dim
    if 'kge_size' not in cfg:
        if 'kge_dim' in cfg:
            cfg['kge_size'] = cfg['kge_dim']
        elif 'relation_dim' in cfg:
            cfg['kge_size'] = cfg['relation_dim']
        else:
            # sensible fallback: tie to embed_size if present, else default 64
            cfg['kge_size'] = cfg.get('embed_size', 64)
    args = argparse.Namespace(**cfg)

    train(args)
