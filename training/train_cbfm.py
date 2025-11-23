from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
from time import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from loaders.cbfm_loader import DataLoaderCBFM
from models.CBFM import CBFM
from utils.log_helper import create_log_id, logging_config
from utils.metrics import calc_metrics_at_k
from utils.model_helper import early_stopping, save_model


def evaluate(model, dataloader, Ks, device):
	model.eval()
	train_user_dict = dataloader.train_user_dict
	test_user_dict = dataloader.test_user_dict

	user_ids = list(test_user_dict.keys())
	limit = dataloader.eval_user_limit
	if len(user_ids) > limit:
		user_ids = user_ids[:limit]

	n_items = dataloader.n_items
	item_ids = torch.arange(n_items, dtype=torch.long, device=device)

	batch_users = torch.LongTensor(user_ids).to(device)
	user_context = dataloader.get_user_context(batch_users)
	if user_context is not None:
		user_context = user_context.to(device)

	with torch.no_grad():
		scores = model(batch_users, item_ids, user_context, mode="predict")
	scores = scores.cpu()

	metrics_raw = calc_metrics_at_k(
		scores,
		train_user_dict,
		test_user_dict,
		np.array(user_ids),
		item_ids.cpu().numpy(),
		Ks,
	)
	metric_names = ["precision", "recall", "f1", "ndcg"]
	metrics_dict = {k: {m: metrics_raw[k][m].mean() for m in metric_names} for k in Ks}
	return metrics_dict


def train(args):
	log_save_id = create_log_id(args.save_dir)
	logging_config(folder=args.save_dir, name=f"log{log_save_id:d}", no_console=False)
	logging.info(args)

	device = torch.device("cuda" if torch.cuda.is_available() and getattr(args, "use_gpu", 1) == 1 else "cpu")

	data = DataLoaderCBFM(args, logging)

	model = CBFM(args, n_users=data.n_users, n_items=data.n_items, context_field_dims=data.context_field_dims)
	model.to(device)
	logging.info(model)

	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	Ks = args.Ks if isinstance(args.Ks, (list, tuple)) else eval(args.Ks)
	k_min = min(Ks)
	k_max = max(Ks)
	stopping_steps = getattr(args, "stopping_steps", 5)

	epoch_list = []
	metrics_list = {k: {"precision": [], "recall": [], "f1": [], "ndcg": []} for k in Ks}
	training_loss = {"epoch": [], "loss": []}
	best_epoch = -1
	best_recall = 0.0

	os.makedirs(args.save_dir, exist_ok=True)

	for epoch in range(1, args.n_epoch + 1):
		t0 = time()
		model.train()
		cf_total_loss = 0.0

		n_cf_batch = data.n_cf_train // data.cf_batch_size + 1
		for iteration in range(1, n_cf_batch + 1):
			batch_user, batch_pos_item, batch_neg_item = data.generate_cf_batch(
				data.train_user_dict, data.cf_batch_size
			)
			batch_user = batch_user.to(device)
			batch_pos_item = batch_pos_item.to(device)
			batch_neg_item = batch_neg_item.to(device)

			user_context = data.get_user_context(batch_user)
			if user_context is not None:
				user_context = user_context.to(device)

			loss = model(
				batch_user,
				batch_pos_item,
				batch_neg_item,
				user_context,
				user_context,
				mode="train_bpr",
			)

			if np.isnan(loss.detach().cpu().numpy()):
				logging.info(
					"ERROR (CBFM Training): Epoch %04d Iter %04d/%04d Loss is nan.",
					epoch,
					iteration,
					n_cf_batch,
				)
				sys.exit()

			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

			cf_total_loss += loss.item()
			if (iteration % getattr(args, "cf_print_every", 100)) == 0 or iteration == n_cf_batch:
				logging.info(
					"CBFM Training: Epoch %04d Iter %04d/%04d | Iter Loss %.4f | Mean %.4f",
					epoch,
					iteration,
					n_cf_batch,
					loss.item(),
					cf_total_loss / iteration,
				)

		epoch_loss = cf_total_loss / max(n_cf_batch, 1)
		training_loss["epoch"].append(epoch)
		training_loss["loss"].append(epoch_loss)
		logging.info(
			"CBFM Training: Epoch %04d Total Iter %04d | Time %.1fs | Mean Loss %.4f",
			epoch,
			n_cf_batch,
			time() - t0,
			epoch_loss,
		)

		if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
			metrics_dict = evaluate(model, data, Ks, device)
			logging.info(
				"CBFM Eval: Epoch %04d | Precision [%.4f, %.4f] Recall [%.4f, %.4f] F1 [%.4f, %.4f] NDCG [%.4f, %.4f]",
				epoch,
				metrics_dict[k_min]["precision"],
				metrics_dict[k_max]["precision"],
				metrics_dict[k_min]["recall"],
				metrics_dict[k_max]["recall"],
				metrics_dict[k_min]["f1"],
				metrics_dict[k_max]["f1"],
				metrics_dict[k_min]["ndcg"],
				metrics_dict[k_max]["ndcg"],
			)

			epoch_list.append(epoch)
			for k in Ks:
				for m in ["precision", "recall", "f1", "ndcg"]:
					metrics_list[k][m].append(metrics_dict[k][m])

			best_recall, should_stop = early_stopping(metrics_list[k_min]["recall"], stopping_steps)
			if metrics_list[k_min]["recall"].index(best_recall) == len(epoch_list) - 1:
				save_model(model, args.save_dir, epoch, best_epoch)
				logging.info("Save CBFM model on epoch %04d!", epoch)
				best_epoch = epoch
			if should_stop:
				break

	training_loss_df = pd.DataFrame(training_loss)
	training_loss_df.to_csv(os.path.join(args.save_dir, "cbfm_training_loss.csv"), index=False)

	metrics_records = []
	for idx, ep in enumerate(epoch_list):
		record = {"epoch_idx": ep}
		for k in Ks:
			for m in ["precision", "recall", "f1", "ndcg"]:
				if idx < len(metrics_list[k][m]):
					record[f"{m}@{k}"] = metrics_list[k][m][idx]
		metrics_records.append(record)
	metrics_df = pd.DataFrame(metrics_records)
	metrics_df.to_csv(os.path.join(args.save_dir, "cbfm_metrics.csv"), index=False)

	if best_epoch != -1 and not metrics_df.empty:
		best_metrics = metrics_df.loc[metrics_df["epoch_idx"] == best_epoch].iloc[0].to_dict()
		logging.info(
			"Best CBFM Eval: Epoch %04d | Precision [%.4f, %.4f] Recall [%.4f, %.4f] F1 [%.4f, %.4f] NDCG [%.4f, %.4f]",
			int(best_metrics["epoch_idx"]),
			best_metrics[f"precision@{k_min}"],
			best_metrics[f"precision@{k_max}"],
			best_metrics[f"recall@{k_min}"],
			best_metrics[f"recall@{k_max}"],
			best_metrics[f"f1@{k_min}"],
			best_metrics[f"f1@{k_max}"],
			best_metrics[f"ndcg@{k_min}"],
			best_metrics[f"ndcg@{k_max}"],
		)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--configs", type=str, required=True, help="Path to YAML config file")
	cli_args = parser.parse_args()
	with open(cli_args.configs, "r") as fh:
		cfg = yaml.safe_load(fh)
	args = argparse.Namespace(**cfg)

	defaults = [
		("embed_dim", 64),
		("lr", 1e-3),
		("cf_batch_size", 1024),
		("test_batch_size", 10000),
		("cf_print_every", 100),
		("n_epoch", 40),
		("evaluate_every", 5),
		("Ks", [20, 40, 60, 80, 100]),
		("save_dir", "trained_model/CBFM"),
		("stopping_steps", 10),
		("use_gpu", 1),
		("use_pretrain", 0),
		("pretrain_embedding_dir", "datasets"),
		("pretrain_model_path", ""),
	]
	for key, value in defaults:
		if not hasattr(args, key):
			setattr(args, key, value)

	os.makedirs(args.save_dir, exist_ok=True)
	train(args)
