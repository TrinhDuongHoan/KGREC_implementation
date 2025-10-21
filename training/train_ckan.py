import os
import sys
import random
from time import time
import yaml
import argparse
import logging
import pathlib

import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ====== models & loaders ======
from models.CKAN import CKAN        
from loaders.ckan_loader import DataLoaderCKAN  
# ====== utils ======
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *


# -------- helper: tile triple-set theo batch --------
def tile_triple_set(ts, n):
    
    H, R, T = ts
    def _tile_list(lst, n):
        outs = []
        for x in lst:
            if x.size(0) == 1:
                outs.append(x.repeat(n, 1))
            else:
                outs.append(x.repeat_interleave(n, dim=0))
        return outs
    return _tile_list(H, n), _tile_list(R, n), _tile_list(T, n)

def evaluate(model, loader, Ks, device, item_chunk_size=1024):
    test_batch_size = loader.test_batch_size
    train_user_dict = loader.train_user_dict
    test_user_dict = loader.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_batches = [user_ids[i:i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]

    n_items = loader.n_items
    all_item_ids = torch.arange(n_items, dtype=torch.long, device=device)

    cf_scores_all_users = []
    metric_names = ['precision', 'recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    with tqdm(total=len(user_batches), desc='Evaluating Iteration') as pbar:
        for ub in user_batches:
            # compute score matrix for this batch: shape [len(ub), n_items]
            batch_scores = []
            for u in ub:
                u_t = torch.tensor([u], dtype=torch.long, device=device)
                # user triple set cho 1 user
                u_ts = loader.build_user_triple_set(u_t)

                scores_u = []
                for i_start in range(0, n_items, item_chunk_size):
                    i_end = min(i_start + item_chunk_size, n_items)
                    items_chunk = all_item_ids[i_start:i_end]
                    i_ts = loader.build_item_triple_set(items_chunk)

                    u_rep = u_t.repeat(items_chunk.size(0))
                    u_ts_rep = tile_triple_set(u_ts, items_chunk.size(0))

                    with torch.no_grad():
                        s = model(u_rep, items_chunk, u_ts_rep, i_ts, mode='predict')  # [C]
                    scores_u.append(s.detach().cpu())

                scores_u = torch.cat(scores_u, dim=0)  # [n_items]
                batch_scores.append(scores_u.unsqueeze(0))  # [1, n_items]

            batch_scores = torch.cat(batch_scores, dim=0)  # [B, n_items]
            batch_scores_cpu = batch_scores.cpu()
            batch_metrics = calc_metrics_at_k(
                batch_scores_cpu,
                train_user_dict,
                test_user_dict,
                np.array(ub, dtype=np.int64),
                np.arange(n_items, dtype=np.int64),
                Ks
            )

            cf_scores_all_users.append(batch_scores_cpu.numpy())
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)

    cf_scores = np.concatenate(cf_scores_all_users, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    return cf_scores, metrics_dict


def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # save_dir
    args.save_dir = (
        f"trained_models/CKAN/{args.data_name}/"
        f"dim{args.embed_dim}_layers{args.n_layer}_{args.agg}_"
        f"lr{args.lr}_pretrain{args.use_pretrain}/"
    )
    os.makedirs(args.save_dir, exist_ok=True)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loader
    data = DataLoaderCKAN(args, logging)

    # model
    model = CKAN(
        n_users=data.n_users,
        n_entity=data.n_users_entities,
        n_relation=data.n_relations,
        dim=args.embed_dim,
        n_layer=args.n_layer,
        agg=args.agg,
        l2=float(args.cf_l2loss_lambda),
    ).to(device)
    logging.info(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    Ks = args.Ks if isinstance(args.Ks, list) else [int(x) for x in str(args.Ks).strip("[]").split(",")]
    k_min, k_max = min(Ks), max(Ks)

    epoch_list = []
    metrics_list = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in Ks}
    training_loss = {'epoch': [], 'cf_loss': [], 'kg_loss': [], 'total_loss': []}  

    best_epoch, best_recall = -1, 0

    for epoch in range(1, args.n_epoch + 1):
        t0 = time()
        model.train()

        # ---- CF training ----
        cf_total_loss = 0.0
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1

        for it in range(1, n_cf_batch + 1):
            users, pos_items, neg_items = data.generate_cf_batch(data.train_user_dict, data.cf_batch_size)
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            # triple-sets
            u_ts_pos = data.build_user_triple_set(users)
            i_ts_pos = data.build_item_triple_set(pos_items)
            u_ts_neg = data.build_user_triple_set(users)
            i_ts_neg = data.build_item_triple_set(neg_items)

            loss = model(users, pos_items, neg_items, u_ts_pos, i_ts_pos, u_ts_neg, i_ts_neg, mode='train_cf')
            if torch.isnan(loss):
                logging.info(f'ERROR (CF Training): Epoch {epoch:04d} Iter {it:04d}/{n_cf_batch:04d} Loss is NaN.')
                sys.exit(1)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cf_total_loss += float(loss.detach().cpu().numpy())
            logging.info(f'CF Training: Epoch {epoch:04d} Iter {it:04d}/{n_cf_batch:04d} Loss {loss:.4f}')

        cf_mean = cf_total_loss / n_cf_batch
        logging.info(f'CF Training: Epoch {epoch:04d} | Total Iter {n_cf_batch} | Mean Loss {cf_mean:.4f}')

        training_loss['epoch'].append(epoch)
        training_loss['cf_loss'].append(cf_mean)
        training_loss['kg_loss'].append(0.0)
        training_loss['total_loss'].append(cf_mean)

        logging.info('Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - t0))

        # ---- Evaluate ----
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            t_eval = time()
            _, metrics_dict = evaluate(model, data, Ks, device)
            logging.info(
                'CF Evaluation: Epoch {:04d} | Time {:.1f}s | '
                'Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                    epoch, time() - t_eval,
                    metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'],
                    metrics_dict[k_min]['recall'],    metrics_dict[k_max]['recall'],
                    metrics_dict[k_min]['ndcg'],      metrics_dict[k_max]['ndcg']
                )
            )

            epoch_list.append(epoch)
            for k in Ks:
                for m in ['precision', 'recall', 'ndcg']:
                    metrics_list[k][m].append(metrics_dict[k][m])

            best_recall, should_stop = early_stopping(metrics_list[k_min]['recall'], args.stopping_steps)
            if should_stop:
                break

            if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:
                save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

    # ---- save training loss ----
    pd.DataFrame(training_loss).to_csv(os.path.join(args.save_dir, "training_loss.csv"), index=False)

    # ---- save metrics CSV ----
    metrics_records = []
    for i, ep in enumerate(epoch_list):
        row = {"epoch_idx": ep}
        for k in Ks:
            for m in ["precision", "recall", "ndcg"]:
                if i < len(metrics_list[k][m]):
                    row[f"{m}@{k}"] = metrics_list[k][m][i]
        metrics_records.append(row)
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(os.path.join(args.save_dir, "metrics.csv"), index=False)

    # ---- print best ----
    if best_epoch != -1 and not metrics_df.empty:
        best_metrics = metrics_df.loc[metrics_df["epoch_idx"] == best_epoch].iloc[0].to_dict()
        logging.info(
            "Best CF Evaluation: Epoch {:04d} | "
            "Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]".format(
                int(best_metrics["epoch_idx"]),
                best_metrics[f"precision@{k_min}"], best_metrics[f"precision@{k_max}"],
                best_metrics[f"recall@{k_min}"],    best_metrics[f"recall@{k_max}"],
                best_metrics[f"ndcg@{k_min}"],      best_metrics[f"ndcg@{k_max}"]
            )
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, required=True, help="Path to YAML config file")
    cli_args = parser.parse_args()

    with open(cli_args.configs, 'r') as f:
        cfg = yaml.safe_load(f)

    def _as_int_list(x, default):
        if isinstance(x, list): return [int(i) for i in x]
        if x is None: return default
        return [int(i) for i in str(x).strip("[]").split(",") if str(i).strip()]


    cfg.setdefault("n_neighbor", 8)

    cfg["Ks"] = _as_int_list(cfg.get("Ks"), [20, 40, 60, 80, 100])


    args = argparse.Namespace(**cfg)
    train(args)
