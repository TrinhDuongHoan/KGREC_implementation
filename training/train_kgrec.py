import os
import sys
import random
from time import time
import time as ttime
import psutil
import yaml
import argparse
import logging


import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.KGRec import KGRec
from loaders.kgrec_loader import DataLoaderKGRec

from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *


def evaluate(model, dataloader, Ks, device):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    cf_scores = []
    metric_names = ['precision', 'recall', 'f1', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    eval_t0 = ttime.perf_counter() 

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)

            with torch.no_grad():
                batch_scores = model(batch_user_ids, item_ids, mode='predict')       # (n_batch_users, n_items)

            batch_scores = batch_scores.cpu()
            batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks)

            cf_scores.append(batch_scores.numpy())
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)

    eval_time_s = max(ttime.perf_counter() - eval_t0, 1e-9)

    # cf_scores = np.concatenate(cf_scores, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
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

    data = DataLoaderKGRec(args, logging)
    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None


    model = KGRec(args, data.n_users, data.n_entities, data.n_relations, data.A_in, user_pre_embed, item_pre_embed)
    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)
    logging.info(model)

    model_n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    model_size_mb = model_param_bytes / (1024.0 * 1024.0)
    logging.info(f"MODEL STATS | trainable_params: {model_n_params:,} | approx_size: {model_size_mb:.2f} MB")

    cf_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    kg_optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_epoch = -1
    best_recall = 0

    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    epoch_list = []
    metrics_list = {k: {'precision': [], 'recall': [], 'f1': [], 'ndcg': []} for k in Ks}
    training_loss = {'epoch': [], 'cf_loss': [], 'kg_loss': [], 'total_loss': []}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_gpu_peak_bytes = 0  
    total_t0 = ttime.perf_counter()
    eval_time_acc = 0.0
    last_eval_time_s = 0.0

    os.makedirs(args.save_dir, exist_ok=True)
    training_stats_path = os.path.join(args.save_dir, "runtime_memory.csv")
    if not os.path.exists(training_stats_path):
        with open(training_stats_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["epoch","iter","phase","wall_time_s","step_time_s","gpu_mem_MB","cpu_mem_MB","batch_size"])

    warm_up_iters = 3

    for epoch in range(1, args.n_epoch + 1):
        time0 = time()
        model.train()

        time1 = time()
        cf_total_loss = 0
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1

        for iter in range(1, n_cf_batch + 1):
            time2 = time()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            iter_wall_start = ttime.perf_counter()

            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(data.train_user_dict, data.cf_batch_size)
            cf_batch_user = cf_batch_user.to(device)
            cf_batch_pos_item = cf_batch_pos_item.to(device)
            cf_batch_neg_item = cf_batch_neg_item.to(device)

            iter_start = ttime.perf_counter()

            cf_batch_loss = model(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item, mode='train_cf')

            if np.isnan(cf_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (CF Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_cf_batch))
                sys.exit()

            cf_batch_loss.backward()
            cf_optimizer.step()
            cf_optimizer.zero_grad()
            cf_total_loss += cf_batch_loss.item()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            
            if (iter % args.cf_print_every) == 0:
                logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_cf_batch, time() - time2, cf_batch_loss.item(), cf_total_loss / iter))
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_cf_batch, time() - time1, cf_total_loss / n_cf_batch))

        # train kg
        time3 = time()
        kg_total_loss = 0
        n_kg_batch = data.n_kg_train // data.kg_batch_size + 1

        for iter in range(1, n_kg_batch + 1):
            time4 = time()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            iter_wall_start = ttime.perf_counter()

            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(data.train_kg_dict, data.kg_batch_size, data.n_users_entities)
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            iter_start = ttime.perf_counter()

            kg_batch_loss = model(kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail, mode='train_kg')

            if np.isnan(kg_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (KG Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_kg_batch))
                sys.exit()

            kg_batch_loss.backward()
            kg_optimizer.step()
            kg_optimizer.zero_grad()
            kg_total_loss += kg_batch_loss.item()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            if (iter % args.kg_print_every) == 0:
                logging.info('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_kg_batch, time() - time4, kg_batch_loss.item(), kg_total_loss / iter))
        logging.info('KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_kg_batch, time() - time3, kg_total_loss / n_kg_batch))
        

        training_loss['epoch'].append(epoch)
        training_loss['cf_loss'].append(cf_total_loss/n_cf_batch)
        training_loss['kg_loss'].append(kg_total_loss/n_kg_batch)
        training_loss['total_loss'].append(cf_total_loss/n_cf_batch + kg_total_loss/n_kg_batch)

        time5 = time()
        h_list = data.h_list.to(device)
        t_list = data.t_list.to(device)
        r_list = data.r_list.to(device)
        relations = list(data.laplacian_dict.keys())
        model(h_list, t_list, r_list, relations, mode='update_att')
        logging.info('Update Attention: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time5))

        logging.info('CF + KG Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

        if torch.cuda.is_available():
            epoch_train_peak = torch.cuda.max_memory_allocated()
            train_gpu_peak_bytes = max(train_gpu_peak_bytes, epoch_train_peak)

        # evaluate cf
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()  # reset so eval peak won't affect training peak

            metrics_dict, eval_time_s = evaluate(model, data, Ks, device)
            eval_time_acc += eval_time_s
            last_eval_time_s = eval_time_s

            logging.info('CF Evaluation: Epoch {:04d} | Eval Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], F1 [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'
                         .format(epoch, eval_time_s,
                                 metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'],
                                 metrics_dict[k_min]['recall'],    metrics_dict[k_max]['recall'],
                                 metrics_dict[k_min]['f1'],        metrics_dict[k_max]['f1'],
                                 metrics_dict[k_min]['ndcg'],      metrics_dict[k_max]['ndcg']))

            # After eval, reset again to start clean for next epoch's training peak accumulation
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            epoch_list.append(epoch)
            for k in Ks:
                for m in ['precision', 'recall', 'f1', 'ndcg']:
                    metrics_list[k][m].append(metrics_dict[k][m])
            best_recall, should_stop = early_stopping(metrics_list[k_min]['recall'], args.stopping_steps)

            if should_stop:
                break

            if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:
                save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

    training_loss_df = pd.DataFrame(training_loss)
    training_loss_path = os.path.join(args.save_dir, "training_loss.csv")
    training_loss_df.to_csv(training_loss_path, index=False)

    # save metrics
    metrics_records = []
    for i, epoch in enumerate(epoch_list):
        row = {"epoch_idx": epoch}
        for k in Ks:
            for m in ["precision", "recall", "f1", "ndcg"]:
                if i < len(metrics_list[k][m]):
                    row[f"{m}@{k}"] = metrics_list[k][m][i]
        metrics_records.append(row)

    metrics_df = pd.DataFrame(metrics_records)
    csv_path = os.path.join(args.save_dir, "kgrect_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)

    # --- In best metrics ---
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


    total_wall_s = ttime.perf_counter() - total_t0
    total_train_s = max(total_wall_s - eval_time_acc, 0.0)
    train_gpu_peak_mb = (train_gpu_peak_bytes / 1024 / 1024) if torch.cuda.is_available() else 0.0

    overall_csv = os.path.join(args.save_dir, "runtime_overall.csv")
    with open(overall_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["total_training_time", "model_size_MB", "model_params", "peak_train_gpu_MB", "inference_time_s"])
        writer.writerow([round(total_train_s, 6), round(model_size_mb, 2), round(model_n_params, 2), round(train_gpu_peak_mb, 2), round(last_eval_time_s, 6)])

    logging.info("RUNTIME SUMMARY | total_training_time: {:.2f} | model_size: {:.2f} | model_params: {:.2f} | peak_train_gpu_MB: {:.1f} | inference_time_s: {:.2f}"
                 .format(total_train_s, model_size_mb, model_n_params, train_gpu_peak_mb, last_eval_time_s))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, required=True, help="Path to YAML config file")
    cli_args = parser.parse_args()

    with open(cli_args.configs, 'r') as f:
        cfg = yaml.safe_load(f)
    
    args = argparse.Namespace(**cfg)

    train(args)
    


