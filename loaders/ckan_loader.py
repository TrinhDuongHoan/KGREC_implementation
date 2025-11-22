import os
import random
import collections

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

from loaders.base_loader import DataLoaderBase


class DataLoaderCKAN(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        self.cf_batch_size = args.cf_batch_size
        self.test_batch_size = args.test_batch_size

        self.n_neighbor = getattr(args, "n_neighbor", 8)
        self.n_layer = getattr(args, "n_layer", 1)
        self.rng = np.random.RandomState(getattr(args, "seed", 2025))

        kg_data = self.load_kg(self.kg_file)
        self.construct_data(kg_data)
        self.print_info(logging)

        self.laplacian_type = args.laplacian_type
        self.create_adjacency_dict()
        self.create_laplacian_dict()  

    def construct_data(self, kg_data: pd.DataFrame):
        n_relations = max(kg_data['r']) + 1

        inverse_kg = kg_data.rename({'h': 't', 't': 'h'}, axis='columns').copy()
        inverse_kg['r'] = inverse_kg['r'] + n_relations
        kg_data = pd.concat([kg_data, inverse_kg], axis=0, ignore_index=True, sort=False)

        kg_data['r'] += 2
        self.n_relations = int(max(kg_data['r']) + 1)
        self.n_entities = int(max(max(kg_data['h']), max(kg_data['t'])) + 1)
        self.n_users_entities = int(self.n_users + self.n_entities)

        self.cf_train_data = (
            np.array([u + self.n_entities for u in self.cf_train_data[0]], dtype=np.int32),
            self.cf_train_data[1].astype(np.int32),
        )
        self.cf_test_data = (
            np.array([u + self.n_entities for u in self.cf_test_data[0]], dtype=np.int32),
            self.cf_test_data[1].astype(np.int32),
        )
        self.train_user_dict = {u + self.n_entities: np.unique(v).astype(np.int32)
                                for u, v in self.train_user_dict.items()}
        self.test_user_dict = {u + self.n_entities: np.unique(v).astype(np.int32)
                               for u, v in self.test_user_dict.items()}

        cf2kg = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg['h'] = self.cf_train_data[0]; cf2kg['t'] = self.cf_train_data[1]

        inv_cf2kg = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        inv_cf2kg['h'] = self.cf_train_data[1]; inv_cf2kg['t'] = self.cf_train_data[0]

        self.kg_train_data = pd.concat([kg_data, cf2kg, inv_cf2kg], ignore_index=True)
        self.n_kg_train = len(self.kg_train_data)

        h_list, t_list, r_list = [], [], []
        self.train_kg_dict = collections.defaultdict(list)        # h -> [(t, r), ...]
        self.train_relation_dict = collections.defaultdict(list)   # r -> [(h, t), ...]

        for _, row in self.kg_train_data.iterrows():
            h, r, t = int(row['h']), int(row['r']), int(row['t'])
            h_list.append(h); t_list.append(t); r_list.append(r)
            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))

        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)

        self.all_items = np.arange(self.n_items, dtype=np.int64)

    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        i = torch.as_tensor(indices, dtype=torch.long)
        v = torch.as_tensor(values, dtype=torch.float32)
        return torch.sparse_coo_tensor(i, v, size=coo.shape).coalesce()

    def create_adjacency_dict(self):
        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            rows = [h for (h, t) in ht_list]
            cols = [t for (h, t) in ht_list]
            vals = [1] * len(rows)
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users_entities, self.n_users_entities))
            self.adjacency_dict[r] = adj

    def create_laplacian_dict(self):
        def symmetric_norm_lap(adj):
            rowsum = np.asarray(adj.sum(axis=1)).flatten()
            d = np.zeros_like(rowsum, dtype=np.float32)
            nz = rowsum > 0
            d[nz] = 1.0 / np.sqrt(rowsum[nz])
            D = sp.diags(d)
            return (D @ adj @ D).tocoo()

        def random_walk_norm_lap(adj):
            rowsum = np.asarray(adj.sum(axis=1)).flatten()
            d = np.zeros_like(rowsum, dtype=np.float32)
            nz = rowsum > 0
            d[nz] = 1.0 / rowsum[nz]
            D = sp.diags(d)
            return (D @ adj).tocoo()

        if self.laplacian_type == 'symmetric':
            norm = symmetric_norm_lap
        elif self.laplacian_type == 'random-walk':
            norm = random_walk_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {r: norm(adj) for r, adj in self.adjacency_dict.items()}
        A_in = sum(self.laplacian_dict.values())
        self.A_in = self.convert_coo2tensor(A_in.tocoo())

    def generate_cf_batch(self, user_dict=None, batch_size=None):
        if user_dict is None:
            user_dict = self.train_user_dict
        if batch_size is None:
            batch_size = self.cf_batch_size

        users = self.rng.choice(list(user_dict.keys()), size=batch_size, replace=True)
        pos, neg = [], []
        for u in users:
            pos_item = self.rng.choice(user_dict[u])
            pos.append(pos_item)
            while True:
                ni = self.rng.randint(0, self.n_items)
                if ni not in user_dict[u]:
                    neg.append(ni); break

        return (torch.LongTensor(users),
                torch.LongTensor(pos),
                torch.LongTensor(neg))

    def _sample_layer(self, seeds_np):

        B = seeds_np.shape[0]
        h = np.zeros((B, self.n_neighbor), dtype=np.int64)
        r = np.zeros((B, self.n_neighbor), dtype=np.int64)
        t = np.zeros((B, self.n_neighbor), dtype=np.int64)

        for i, e in enumerate(seeds_np):
            neigh = self.train_kg_dict.get(int(e), None)
            if not neigh:
                continue
            idx = self.rng.randint(0, len(neigh), size=self.n_neighbor)
            for j, k in enumerate(idx):
                _t, _r = neigh[k]        
                h[i, j] = e
                r[i, j] = _r
                t[i, j] = _t
        return h, r, t

    def _build_triple_set(self, seeds_tensor):

        device = seeds_tensor.device
        seeds = seeds_tensor.detach().cpu().numpy()
        H_list, R_list, T_list = [], [], []

        for _ in range(self.n_layer):
            h, r, t = self._sample_layer(seeds)
            H_list.append(torch.from_numpy(h).to(device))
            R_list.append(torch.from_numpy(r).to(device))
            T_list.append(torch.from_numpy(t).to(device))
            seeds = t[:, 0]

        
        if not H_list:  
            h0 = seeds_tensor.unsqueeze(1).repeat(1, self.n_neighbor)
            r0 = torch.zeros_like(h0)
            t0 = h0.clone()
            H_list, R_list, T_list = [h0], [r0], [t0]

        return (H_list, R_list, T_list)

    def build_user_triple_set(self, users: torch.LongTensor):
        seeds = []
        for u in users.tolist():
            items = self.train_user_dict[u]
            seeds.append(self.rng.choice(items))
        seeds = torch.LongTensor(seeds).to(users.device)
        return self._build_triple_set(seeds)

    def build_item_triple_set(self, items: torch.LongTensor):
        return self._build_triple_set(items)

    def print_info(self, logging):
        logging.info('n_users:           %d' % self.n_users)
        logging.info('n_items:           %d' % self.n_items)
        logging.info('n_entities:        %d' % self.n_entities)
        logging.info('n_users_entities:  %d' % self.n_users_entities)
        logging.info('n_relations:       %d' % self.n_relations)
        logging.info('n_h_list:          %d' % len(self.h_list))
        logging.info('n_t_list:          %d' % len(self.t_list))
        logging.info('n_r_list:          %d' % len(self.r_list))
        logging.info('n_cf_train:        %d' % self.n_cf_train)
        logging.info('n_cf_test:         %d' % self.n_cf_test)
        logging.info('n_kg_train:        %d' % self.n_kg_train)
