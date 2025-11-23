import os
import sys
import random
from time import time
import time as ttime
import yaml
import argparse
import logging
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from loaders.mcrec_loader import DataLoaderMCRec
from models.MCRec import MCRec
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *


def evaluate(model, dataloader, Ks, device):
	model.eval()
	train_user_dict = dataloader.train_user_dict
	test_user_dict = dataloader.test_user_dict

	user_ids = list(test_user_dict.keys())
	limit = getattr(dataloader.args, 'eval_user_limit', 1000)
	if len(user_ids) > limit:
		user_ids = user_ids[:limit]

	n_items = dataloader.n_items
	item_ids = torch.arange(n_items, dtype=torch.long, device=device)

	cf_scores = []
	metric_names = ['precision', 'recall', 'f1', 'ndcg']
	metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

	eval_t0 = ttime.perf_counter()

	with tqdm(total=len(user_ids), desc='Evaluating Iteration') as pbar:
		for u in user_ids:
			user_tensor = torch.full((n_items,), u, dtype=torch.long, device=device)
			def zp(pn, ts):
				return torch.zeros(n_items, pn, ts, dataloader.fea_size, device=device)
			umtm = zp(dataloader.umtm_path_num, dataloader.umtm_timestamp)
			umum = zp(dataloader.umum_path_num, dataloader.umum_timestamp)
			umtmum = zp(dataloader.umtmum_path_num, dataloader.umtmum_timestamp)
			uuum = zp(dataloader.uuum_path_num, dataloader.uuum_timestamp)

			with torch.no_grad():
				scores = model(user_tensor, item_ids, umtm, umum, umtmum, uuum).view(-1)

			scores = scores.cpu()

			train_pos = train_user_dict.get(u, [])
			if train_pos:
				scores[train_pos] = -1e9

			cf_scores.append(scores.unsqueeze(0))
			pbar.update(1)

	eval_time_s = max(ttime.perf_counter() - eval_t0, 1e-9)
	cf_scores = torch.cat(cf_scores, dim=0)

	cf_scores_np = cf_scores.clone()
	metrics_raw = calc_metrics_at_k(cf_scores_np, train_user_dict, test_user_dict, np.array(user_ids), item_ids.cpu().numpy(), Ks)
	for k in Ks:
		for m in metric_names:
			metrics_dict[k][m] = metrics_raw[k][m].mean()

	return metrics_dict, eval_time_s


def train(args):
	log_save_id = create_log_id(args.save_dir)
	logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
	logging.info(args)

	device = torch.device("cuda" if torch.cuda.is_available() and getattr(args, 'use_gpu', 1) == 1 else "cpu")

	data = DataLoaderMCRec(args, logging)

	model = MCRec(
		n_users=data.n_users,
		n_items=data.n_items,
		path_nums=[data.umtm_path_num, data.umum_path_num, data.umtmum_path_num, data.uuum_path_num],
		timestamps=[data.umtm_timestamp, data.umum_timestamp, data.umtmum_timestamp, data.uuum_timestamp],
		feat_len=data.fea_size,
		latent_dim=getattr(args, 'embed_dim', 64),
		mlp_layers=getattr(args, 'mlp_layers', [512, 256, 128, 64])
	)

	model.to(device)
	logging.info(model)

	model_n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	model_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
	model_size_mb = model_param_bytes / (1024.0 * 1024.0)
	logging.info(f"MODEL STATS | trainable_params: {model_n_params:,} | approx_size: {model_size_mb:.2f} MB")

	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	bce = nn.BCELoss()

	best_epoch = -1
	best_recall = 0

	Ks = args.Ks if isinstance(args.Ks, (list, tuple)) else eval(args.Ks)
	k_min = min(Ks)
	k_max = max(Ks)

	epoch_list = []
	metrics_list = {k: {'precision': [], 'recall': [], 'f1': [], 'ndcg': []} for k in Ks}
	training_loss = {'epoch': [], 'loss': []}

	if torch.cuda.is_available():
		torch.cuda.reset_peak_memory_stats()
	train_gpu_peak_bytes = 0
	total_t0 = ttime.perf_counter()
	eval_time_acc = 0.0
	last_eval_time_s = 0.0

	os.makedirs(args.save_dir, exist_ok=True)

	stopping_steps = getattr(args, 'stopping_steps', 5)

	for epoch in range(1, args.n_epoch + 1):
		time0 = time()
		model.train()

		train_pairs = list(zip(data.cf_train_data[0], data.cf_train_data[1]))
		n_cf_batch = len(train_pairs) // data.cf_batch_size + 1
		batch_iter = data.generate_mcrec_batch(train_pairs, args.num_neg, data.cf_batch_size)

		cf_total_loss = 0

		for iter in range(1, n_cf_batch + 1):
			if torch.cuda.is_available():
				torch.cuda.reset_peak_memory_stats()
			try:
				batch = next(batch_iter)
			except StopIteration:
				break
			user, item, umtm, umum, umtmum, uuum, labels = [x.to(device) for x in batch]
			preds = model(user, item, umtm, umum, umtmum, uuum, mode='train').view(-1)
			cf_batch_loss = bce(preds, labels)
			if np.isnan(cf_batch_loss.detach().cpu().numpy()):
				logging.info(f'ERROR (MCRec Training): Epoch {epoch:04d} Iter {iter:04d}/{n_cf_batch:04d} Loss is nan.')
				sys.exit()
			cf_batch_loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			cf_total_loss += cf_batch_loss.item()
			if torch.cuda.is_available():
				torch.cuda.synchronize()
			if (iter % getattr(args, 'cf_print_every', 100)) == 0 or iter == n_cf_batch:
				logging.info('MCRec Training: Epoch {:04d} Iter {:04d}/{:04d} | Iter Loss {:.4f} | Mean {:.4f}'.format(
					epoch, iter, n_cf_batch, cf_batch_loss.item(), cf_total_loss / iter))

		logging.info('MCRec Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Mean Loss {:.4f}'.format(
			epoch, n_cf_batch, time() - time0, cf_total_loss / max(n_cf_batch,1)))

		training_loss['epoch'].append(epoch)
		training_loss['loss'].append(cf_total_loss / max(n_cf_batch,1))

		if torch.cuda.is_available():
			epoch_train_peak = torch.cuda.max_memory_allocated()
			train_gpu_peak_bytes = max(train_gpu_peak_bytes, epoch_train_peak)

		if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
			if torch.cuda.is_available():
				torch.cuda.reset_peak_memory_stats()
			metrics_dict, eval_time_s = evaluate(model, data, Ks, device)
			eval_time_acc += eval_time_s
			last_eval_time_s = eval_time_s
			logging.info('MCRec Evaluation: Epoch {:04d} | Eval Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], F1 [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
				epoch, eval_time_s,
				metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'],
				metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'],
				metrics_dict[k_min]['f1'], metrics_dict[k_max]['f1'],
				metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))

			epoch_list.append(epoch)
			metric_names_loop = ['precision','recall','f1','ndcg']
			for k in Ks:
				for m in metric_names_loop:
					metrics_list[k][m].append(metrics_dict[k][m])
			best_recall, should_stop = early_stopping(metrics_list[k_min]['recall'], stopping_steps)
			if should_stop:
				break
			if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:
				save_model(model, args.save_dir, epoch, best_epoch)
				logging.info('Save model on epoch {:04d}!'.format(epoch))
				best_epoch = epoch

	training_loss_df = pd.DataFrame(training_loss)
	training_loss_path = os.path.join(args.save_dir, 'training_loss.csv')
	training_loss_df.to_csv(training_loss_path, index=False)

	metrics_records = []
	for i, epoch in enumerate(epoch_list):
		row = {'epoch_idx': epoch}
		for k in Ks:
			for m in ['precision','recall','f1','ndcg']:
				if i < len(metrics_list[k][m]):
					row[f'{m}@{k}'] = metrics_list[k][m][i]
		metrics_records.append(row)
	metrics_df = pd.DataFrame(metrics_records)
	csv_path = os.path.join(args.save_dir, 'mcrec_metrics.csv')
	metrics_df.to_csv(csv_path, index=False)

	if best_epoch != -1:
		best_metrics = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()
		logging.info('Best MCRec Eval: Epoch {:04d} | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], F1 [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
			int(best_metrics['epoch_idx']),
			best_metrics[f'precision@{k_min}'], best_metrics[f'precision@{k_max}'],
			best_metrics[f'recall@{k_min}'], best_metrics[f'recall@{k_max}'],
			best_metrics[f'f1@{k_min}'], best_metrics[f'f1@{k_max}'],
			best_metrics[f'ndcg@{k_min}'], best_metrics[f'ndcg@{k_max}']))

	total_wall_s = ttime.perf_counter() - total_t0
	total_train_s = max(total_wall_s - eval_time_acc, 0.0)
	train_gpu_peak_mb = (train_gpu_peak_bytes / 1024 / 1024) if torch.cuda.is_available() else 0.0

	overall_csv = os.path.join(args.save_dir, 'runtime_overall.csv')
	with open(overall_csv, 'w', newline='') as fh:
		writer = csv.writer(fh)
		writer.writerow(['total_training_time','model_size_MB','model_params','peak_train_gpu_MB','inference_time_s'])
		writer.writerow([round(total_train_s,6), round(model_size_mb,2), round(model_n_params,2), round(train_gpu_peak_mb,2), round(last_eval_time_s,6)])
	logging.info('RUNTIME SUMMARY | total_training_time: {:.2f} | model_size: {:.2f} | model_params: {:.2f} | peak_train_gpu_MB: {:.1f} | inference_time_s: {:.2f}'.format(
		total_train_s, model_size_mb, model_n_params, train_gpu_peak_mb, last_eval_time_s))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--configs', type=str, required=True, help='Path to YAML config file')
	cli_args = parser.parse_args()
	with open(cli_args.configs, 'r') as f:
		cfg = yaml.safe_load(f)
	args = argparse.Namespace(**cfg)
	for k,v in [('batch_size',256),('num_neg',4),('n_epoch',10),('evaluate_every',1),('lr',0.001),('Ks',[10]),('save_dir','trained_model/MCRec'),('stopping_steps',5)]:
		if not hasattr(args,k):
			setattr(args,k,v)
	train(args)
