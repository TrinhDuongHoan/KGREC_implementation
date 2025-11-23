import os
import sys
import argparse
import yaml
import logging
from time import time
import time as ttime
import pathlib
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from models.AMIE import AMIE
from loaders.base_loader import DataLoaderBase  
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

	batch_users = torch.LongTensor(user_ids).to(device)
	with torch.no_grad():
		scores = model(batch_users, item_ids, mode='predict')  
	scores = scores.cpu()

	metrics_raw = calc_metrics_at_k(scores, train_user_dict, test_user_dict, np.array(user_ids), item_ids.cpu().numpy(), Ks)
	metric_names = ['precision','recall','f1','ndcg']
	metrics_dict = {k: {m: metrics_raw[k][m].mean() for m in metric_names} for k in Ks}
	return metrics_dict, 0.0  # evaluation time not tracked precisely here


def train(args):
	log_save_id = create_log_id(args.save_dir)
	logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
	logging.info(args)

	device = torch.device('cuda' if torch.cuda.is_available() and getattr(args,'use_gpu',1)==1 else 'cpu')

	data = DataLoaderBase(args, logging)

	model = AMIE(args, n_users=data.n_users, n_items=data.n_items)
	model.to(device)
	logging.info(model)

	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	Ks = args.Ks if isinstance(args.Ks,(list,tuple)) else eval(args.Ks)
	k_min = min(Ks); k_max = max(Ks)
	stopping_steps = getattr(args,'stopping_steps',5)

	epoch_list = []
	metrics_list = {k:{'precision':[], 'recall':[], 'f1':[], 'ndcg':[]} for k in Ks}
	training_loss = {'epoch':[], 'loss':[]}
	best_epoch = -1
	best_recall = 0

	for epoch in range(1, args.n_epoch + 1):
		t0 = time()
		model.train()
		cf_total_loss = 0

		n_cf_batch = data.n_cf_train // args.cf_batch_size + 1
		for iter in range(1, n_cf_batch + 1):
			batch_user, batch_pos_item, batch_neg_item = data.generate_cf_batch(data.train_user_dict, args.cf_batch_size)
			batch_user = batch_user.to(device)
			batch_pos_item = batch_pos_item.to(device)
			batch_neg_item = batch_neg_item.to(device)
			loss = model(batch_user, batch_pos_item, batch_neg_item, mode='train_cf')
			if np.isnan(loss.detach().cpu().numpy()):
				logging.info(f'ERROR (AMIE Training): Epoch {epoch:04d} Iter {iter:04d}/{n_cf_batch:04d} Loss is nan.')
				sys.exit()
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			cf_total_loss += loss.item()
			if (iter % getattr(args,'cf_print_every',100)) == 0 or iter == n_cf_batch:
				logging.info('AMIE Training: Epoch {:04d} Iter {:04d}/{:04d} | Iter Loss {:.4f} | Mean {:.4f}'.format(epoch, iter, n_cf_batch, loss.item(), cf_total_loss/iter))
		logging.info('AMIE Training: Epoch {:04d} Total Iter {:04d} | Time {:.1f}s | Mean Loss {:.4f}'.format(epoch, n_cf_batch, time()-t0, cf_total_loss/max(n_cf_batch,1)))
		training_loss['epoch'].append(epoch)
		training_loss['loss'].append(cf_total_loss/max(n_cf_batch,1))

		if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
			metrics_dict, _ = evaluate(model, data, Ks, device)
			logging.info('AMIE Eval: Epoch {:04d} | Precision [{:.4f}, {:.4f}] Recall [{:.4f}, {:.4f}] F1 [{:.4f}, {:.4f}] NDCG [{:.4f}, {:.4f}]'.format(
				epoch,
				metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'],
				metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'],
				metrics_dict[k_min]['f1'], metrics_dict[k_max]['f1'],
				metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))
			epoch_list.append(epoch)
			for k in Ks:
				for m in ['precision','recall','f1','ndcg']:
					metrics_list[k][m].append(metrics_dict[k][m])
			best_recall, should_stop = early_stopping(metrics_list[k_min]['recall'], stopping_steps)
			if should_stop:
				break
			if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:
				save_model(model, args.save_dir, epoch, best_epoch)
				logging.info('Save AMIE model on epoch {:04d}!'.format(epoch))
				best_epoch = epoch

	training_loss_df = pd.DataFrame(training_loss)
	training_loss_df.to_csv(os.path.join(args.save_dir,'amie_training_loss.csv'), index=False)
	metrics_records = []
	for i, ep in enumerate(epoch_list):
		row = {'epoch_idx':ep}
		for k in Ks:
			for m in ['precision','recall','f1','ndcg']:
				if i < len(metrics_list[k][m]):
					row[f'{m}@{k}'] = metrics_list[k][m][i]
		metrics_records.append(row)
	metrics_df = pd.DataFrame(metrics_records)
	metrics_df.to_csv(os.path.join(args.save_dir,'amie_metrics.csv'), index=False)
	if best_epoch != -1:
		best_metrics = metrics_df.loc[metrics_df['epoch_idx']==best_epoch].iloc[0].to_dict()
		logging.info('Best AMIE Eval: Epoch {:04d} | Precision [{:.4f}, {:.4f}] Recall [{:.4f}, {:.4f}] F1 [{:.4f}, {:.4f}] NDCG [{:.4f}, {:.4f}]'.format(
			int(best_metrics['epoch_idx']),
			best_metrics[f'precision@{k_min}'], best_metrics[f'precision@{k_max}'],
			best_metrics[f'recall@{k_min}'], best_metrics[f'recall@{k_max}'],
			best_metrics[f'f1@{k_min}'], best_metrics[f'f1@{k_max}'],
			best_metrics[f'ndcg@{k_min}'], best_metrics[f'ndcg@{k_max}']))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--configs', type=str, required=True, help='Path to YAML config file')
	cli_args = parser.parse_args()
	with open(cli_args.configs,'r') as f:
		cfg = yaml.safe_load(f)
	args = argparse.Namespace(**cfg)
	for k,v in [('embed_dim',64),('n_interest',4),('interest_dim',64),('lr',0.001),('cf_batch_size',256),('cf_print_every',100),('n_epoch',10),('evaluate_every',1),('Ks',[10]),('save_dir','trained_model/AMIE'),('stopping_steps',5)]:
		if not hasattr(args,k):
			setattr(args,k,v)
	os.makedirs(args.save_dir, exist_ok=True)
	train(args)
