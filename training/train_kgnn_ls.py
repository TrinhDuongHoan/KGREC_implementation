import os
import sys
import random
from time import time
import time as ttime
import yaml
import argparse
import logging
import csv

import pandas as pd
from tqdm import tqdm
import torch
import torch.optim as optim
import numpy as np
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.KGNN_LS import KGNN_LS  
from loaders.kgnn_ls_loader import DataLoaderKGNNLS 

from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *

def ls_schedule(epoch):
    if epoch <= 3:  return 0.0
    if epoch <= 6:  return 0.05
    return 0.1

@torch.no_grad()
def evaluate(model, loader, Ks, device, user_bs=256, item_chunk=4096, use_fp16=True):
    eval_t0 = ttime.perf_counter()
    train_user_dict = loader.train_user_dict
    test_user_dict  = loader.test_user_dict

    # precompute item cache  (1 lần)
    model.build_item_cache(device, n_items=loader.n_items, chunk=item_chunk)

    user_ids = list(test_user_dict.keys())
    metric_names = ['precision', 'recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    for s in range(0, len(user_ids), user_bs):
        e = min(s + user_bs, len(user_ids))
        u_batch = torch.tensor(user_ids[s:e], dtype=torch.long, device=device)

        scores = model.predict_from_cache(u_batch)          # [B, I], float32
        if use_fp16:
            scores = scores.half()                          # giảm 1/2 RAM
        # chuyển sang numpy theo batch, KHÔNG tích lũy cf_scores
        batch_metrics = calc_metrics_at_k(
            scores.cpu(),
            train_user_dict, test_user_dict,
            u_batch.cpu().numpy(),
            np.arange(loader.n_items, dtype=np.int64),
            Ks
        )
        for k in Ks:
            for m in metric_names:
                metrics_dict[k][m].append(batch_metrics[k][m])

        del scores  # giải phóng

    # gộp trung bình
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    eval_time_s = max(ttime.perf_counter() - eval_t0, 1e-9)
    return None, metrics_dict, eval_time_s  # không trả cf_scores để tiết kiệm RAM


def train(args):
    # seed
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = DataLoaderKGNNLS(args, logging)

    model = KGNN_LS(
        args,
        n_user=data.n_users,
        n_entity=data.n_entities,          # items + entities khác
        n_relation=data.n_relations,
        adj_entity=data.adj_entity,        # shape [n_entity, K^1 + ...] như bạn đang dùng
        adj_relation=data.adj_relation,
        user_pos_items=data.train_user_dict,  # dict u -> LongTensor(items)
        n_items=data.n_items,             
        user_pre_embed=(data.user_pre_embed if args.use_pretrain else None),
        item_pre_embed=(data.item_pre_embed if args.use_pretrain else None),
        use_pretrain=args.use_pretrain,
    ).to(device)

    logging.info(model)

    # --- model stats: number of trainable params and approximate model size (MB) ---
    model_n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    model_size_mb = model_param_bytes / (1024.0 * 1024.0)
    logging.info(f"MODEL STATS | trainable_params: {model_n_params:,} | approx_size: {model_size_mb:.2f} MB")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    Ks = eval(args.Ks); k_min, k_max = min(Ks), max(Ks)
    epoch_list = []; metrics_list = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in Ks}
    training_loss = {'epoch': [], 'cf_loss': [], 'kg_loss': [], 'total_loss': []}
    best_epoch = -1

    # Overall trackers
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_gpu_peak_bytes = 0
    overall_t0 = ttime.perf_counter()
    eval_time_acc = 0.0
    last_eval_time_s = 0.0

    for epoch in range(1, args.n_epoch + 1):
        time0 = time()
        model.train()
        ls_w = ls_schedule(epoch)
        cf_total_loss = 0.0
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1

        for it in range(1, n_cf_batch + 1):
            users, pos_i, neg_i = data.generate_cf_batch(data.train_user_dict, data.cf_batch_size)
            users = users.to(device); pos_i = pos_i.to(device); neg_i = neg_i.to(device)

            loss = model(users, pos_i, mode='train_cf', neg_items=neg_i,
                         ls_weight_eff=ls_w, ls_subsample=256)
            if torch.isnan(loss):
                logging.info(f'ERROR (CF Training): Epoch {epoch:04d} Iter {it:04d}/{n_cf_batch:04d} Loss is nan.')
                sys.exit(1)

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            cf_total_loss += float(loss.item())

            if (it % args.cf_print_every) == 0:
                logging.info('CF Training: Epoch {:04d} Iter {:04d}/{:04d} | Time {:.1f}s | Iter Loss {:.4f} | Mean {:.4f}'.format(
                    epoch, it, n_cf_batch, time() - time0, loss.item(), cf_total_loss / it))

        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Mean Loss {:.4f}'.format(
            epoch, n_cf_batch, time() - time0, cf_total_loss / max(n_cf_batch, 1)))

        # no KG update
        training_loss['epoch'].append(epoch)
        cf_mean = cf_total_loss / max(n_cf_batch, 1)
        training_loss['cf_loss'].append(cf_mean)
        training_loss['kg_loss'].append(0.0)
        training_loss['total_loss'].append(cf_mean)

        # updatemode=_att noop
        model(None, None, mode='update_att')

        # snapshot training GPU peak up to now
        if torch.cuda.is_available():
            train_gpu_peak_bytes = max(train_gpu_peak_bytes, torch.cuda.max_memory_allocated())

        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            # snapshot training peak and reset to isolate eval
            if torch.cuda.is_available():
                train_gpu_peak_bytes = max(train_gpu_peak_bytes, torch.cuda.max_memory_allocated())
                torch.cuda.reset_peak_memory_stats()
            time6 = time()
            _, metrics_dict, eval_time_s = evaluate(model, data, Ks, device)
            last_eval_time_s = eval_time_s
            eval_time_acc += eval_time_s
            logging.info('CF Evaluation: Epoch {:04d} | Time {:.1f}s | P [{:.4f}, {:.4f}], R [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                epoch, eval_time_s,
                metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'],
                metrics_dict[k_min]['recall'],    metrics_dict[k_max]['recall'],
                metrics_dict[k_min]['ndcg'],      metrics_dict[k_max]['ndcg']
            ))

            epoch_list.append(epoch)
            for k in Ks:
                for m in ['precision', 'recall', 'ndcg']:
                    metrics_list[k][m].append(metrics_dict[k][m])

            best_recall, should_stop = early_stopping(metrics_list[k_min]['recall'], args.stopping_steps)
            if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:
                save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch
            if should_stop:
                break

    # save loss
    pd.DataFrame(training_loss).to_csv(os.path.join(args.save_dir, "training_loss.csv"), index=False)

    # save metrics
    records = []
    for i, ep in enumerate(epoch_list):
        row = {"epoch_idx": ep}
        for k in Ks:
            for m in ["precision", "recall", "ndcg"]:
                if i < len(metrics_list[k][m]):
                    row[f"{m}@{k}"] = metrics_list[k][m][i]
        records.append(row)
    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(os.path.join(args.save_dir, "metrics.csv"), index=False)

    # best metrics
    if best_epoch == -1 and len(metrics_df) > 0 and f"recall@{k_min}" in metrics_df.columns:
        recalls = metrics_df[f"recall@{k_min}"].to_numpy()
        best_idx = int(np.nanargmax(recalls))
        best_epoch = int(metrics_df.iloc[best_idx]["epoch_idx"])
    row = metrics_df.loc[metrics_df["epoch_idx"] == best_epoch]
    if not row.empty:
        bm = row.iloc[0].to_dict()
        logging.info("Best CF Evaluation: Epoch {:04d} | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]".format(
            int(bm["epoch_idx"]),
            bm.get(f"precision@{k_min}", float('nan')), bm.get(f"precision@{k_max}", float('nan')),
            bm.get(f"recall@{k_min}", float('nan')),    bm.get(f"recall@{k_max}", float('nan')),
            bm.get(f"ndcg@{k_min}", float('nan')),      bm.get(f"ndcg@{k_max}", float('nan')),
        ))

    # Overall summary
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
    logging.info(
        "RUNTIME SUMMARY | train_total_s: {:.2f} | peak_train_gpu_MB: {:.1f} | inference_time_s: {:.2f} | params: {} | model_size_MB: {:.2f}".format(
            total_train_s, train_gpu_peak_mb, last_eval_time_s, model_n_params, model_size_mb
        )
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, required=True)
    cli_args = parser.parse_args()
    with open(cli_args.configs, 'r') as f:
        cfg = yaml.safe_load(f)
    args = argparse.Namespace(**cfg)
    train(args)
