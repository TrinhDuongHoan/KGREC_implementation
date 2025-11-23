import os
import numpy as np
import torch
from .base_loader import DataLoaderBase


class DataLoaderMCRec(DataLoaderBase):

	def __init__(self, args, logging):
		if not hasattr(args, 'use_pretrain'):
			setattr(args, 'use_pretrain', 0)
		if not hasattr(args, 'pretrain_embedding_dir'):
			setattr(args, 'pretrain_embedding_dir', '')
		if not hasattr(args, 'seed'):
			setattr(args, 'seed', 42)

		super().__init__(args, logging)

		self.cf_batch_size = getattr(args, 'cf_batch_size', getattr(args, 'batch_size', 256))
		self.test_batch_size = getattr(args, 'test_batch_size', 1000)
		self.cf_print_every = getattr(args, 'cf_print_every', 100)

		self.umtm_path_num = getattr(args, 'umtm_path_num', 3)
		self.umum_path_num = getattr(args, 'umum_path_num', 2)
		self.umtmum_path_num = getattr(args, 'umtmum_path_num', 2)
		self.uuum_path_num = getattr(args, 'uuum_path_num', 1)

		self.umtm_timestamp = getattr(args, 'umtm_timestamp', 10)
		self.umum_timestamp = getattr(args, 'umum_timestamp', 8)
		self.umtmum_timestamp = getattr(args, 'umtmum_timestamp', 6)
		self.uuum_timestamp = getattr(args, 'uuum_timestamp', 5)

		self.fea_size = getattr(args, 'mcrec_feature_size', 64)

		rng = np.random.default_rng(seed=getattr(args, 'seed', 42))
		self.user_feature = rng.standard_normal((self.n_users, self.fea_size)).astype(np.float32)
		self.item_feature = rng.standard_normal((self.n_items, self.fea_size)).astype(np.float32)
		self.n_types = getattr(args, 'n_types', 50)
		self.type_feature = rng.standard_normal((self.n_types, self.fea_size)).astype(np.float32)

		self.path_umtm = {}
		self.path_umum = {}
		self.path_umtmum = {}
		self.path_uuum = {}
		self._build_dummy_paths(rng)

	def _build_dummy_paths(self, rng):
		users = self.cf_train_data[0]
		items = self.cf_train_data[1]
		for u, i in zip(users, items):
			if (u, i) not in self.path_umtm:
				self.path_umtm[(u, i)] = self._rand_paths(rng, self.umtm_path_num, self.umtm_timestamp)
			if (u, i) not in self.path_umum:
				self.path_umum[(u, i)] = self._rand_paths(rng, self.umum_path_num, self.umum_timestamp)
			if (u, i) not in self.path_umtmum:
				self.path_umtmum[(u, i)] = self._rand_paths(rng, self.umtmum_path_num, self.umtmum_timestamp)
			if (u, i) not in self.path_uuum:
				self.path_uuum[(u, i)] = self._rand_paths(rng, self.uuum_path_num, self.uuum_timestamp)

	def _rand_paths(self, rng, path_num, timestamp_len):
		paths = []
		for _ in range(path_num):
			seq = []
			for _ in range(timestamp_len):
				type_category = rng.integers(1, 4)
				if type_category == 1:
					idx = rng.integers(0, self.n_users)
				elif type_category == 2:
					idx = rng.integers(0, self.n_items)
				else:
					idx = rng.integers(0, self.n_types)
				seq.append((type_category, idx))
			paths.append(seq)
		return paths

	def generate_mcrec_batch(self, train_list, num_negatives, batch_size):
		data_size = len(train_list)
		indices = np.arange(data_size)
		np.random.shuffle(indices)

		for start in range(0, data_size, batch_size):
			end = min(start + batch_size, data_size)
			batch_pairs = [train_list[idx] for idx in indices[start:end]]

			k = 0
			total = len(batch_pairs) * (num_negatives + 1)
			path_nums = [self.umtm_path_num, self.umum_path_num, self.umtmum_path_num, self.uuum_path_num]
			timestamps = [self.umtm_timestamp, self.umum_timestamp, self.umtmum_timestamp, self.uuum_timestamp]

			user_input = np.zeros((total,), dtype=np.int64)
			item_input = np.zeros((total,), dtype=np.int64)
			umtm_input = np.zeros((total, path_nums[0], timestamps[0], self.fea_size), dtype=np.float32)
			umum_input = np.zeros((total, path_nums[1], timestamps[1], self.fea_size), dtype=np.float32)
			umtmum_input = np.zeros((total, path_nums[2], timestamps[2], self.fea_size), dtype=np.float32)
			uuum_input = np.zeros((total, path_nums[3], timestamps[3], self.fea_size), dtype=np.float32)
			labels = np.zeros((total,), dtype=np.float32)

			for (u, i) in batch_pairs:
				user_input[k] = u
				item_input[k] = i
				self._fill_paths((u, i), k, umtm_input, umum_input, umtmum_input, uuum_input)
				labels[k] = 1.0
				k += 1
				for _ in range(num_negatives):
					j = np.random.randint(0, self.n_items)
					while j in self.train_user_dict[u]:
						j = np.random.randint(0, self.n_items)
					user_input[k] = u
					item_input[k] = j
					self._fill_paths((u, j), k, umtm_input, umum_input, umtmum_input, uuum_input)
					labels[k] = 0.0
					k += 1

			yield (
				torch.from_numpy(user_input),
				torch.from_numpy(item_input),
				torch.from_numpy(umtm_input),
				torch.from_numpy(umum_input),
				torch.from_numpy(umtmum_input),
				torch.from_numpy(uuum_input),
				torch.from_numpy(labels),
			)

	def _fill_paths(self, pair, k, umtm_input, umum_input, umtmum_input, uuum_input):
		(u, i) = pair
		if (u, i) not in self.path_umtm:
			rng = np.random.default_rng()
			self.path_umtm[(u, i)] = self._rand_paths(rng, self.umtm_path_num, self.umtm_timestamp)
			self.path_umum[(u, i)] = self._rand_paths(rng, self.umum_path_num, self.umum_timestamp)
			self.path_umtmum[(u, i)] = self._rand_paths(rng, self.umtmum_path_num, self.umtmum_timestamp)
			self.path_uuum[(u, i)] = self._rand_paths(rng, self.uuum_path_num, self.uuum_timestamp)

		def assign(path_dict, arr, path_num, timestamp_len):
			paths = path_dict[(u, i)]
			for p_i in range(path_num):
				seq = paths[p_i]
				for p_j in range(timestamp_len):
					type_id, index = seq[p_j]
					if type_id == 1:
						arr[k, p_i, p_j] = self.user_feature[index]
					elif type_id == 2:
						arr[k, p_i, p_j] = self.item_feature[index]
					else:
						arr[k, p_i, p_j] = self.type_feature[index]

		assign(self.path_umtm, umtm_input, self.umtm_path_num, self.umtm_timestamp)
		assign(self.path_umum, umum_input, self.umum_path_num, self.umum_timestamp)
		assign(self.path_umtmum, umtmum_input, self.umtmum_path_num, self.umtmum_timestamp)
		assign(self.path_uuum, uuum_input, self.uuum_path_num, self.uuum_timestamp)

