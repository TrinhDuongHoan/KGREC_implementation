import os
import numpy as np
import torch
import pandas as pd
import collections

from loaders.base_loader import DataLoaderBase

class DataLoaderKGNNLS(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        self.cf_batch_size = args.batch_size
        self.kg_batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.neighbor_sample_size = int(args.neighbor_sample_size)

        kg_df = self.load_kg(self.kg_file)
        self._construct_from_kg(kg_df, logging)

        self.h_list = torch.LongTensor([])
        self.t_list = torch.LongTensor([])
        self.r_list = torch.LongTensor([])
        self.laplacian_dict = {}
        self.A_in = torch.sparse_coo_tensor(size=(self.n_users + self.n_entities,
                                                  self.n_users + self.n_entities))

    def _construct_from_kg(self, kg_df: pd.DataFrame, logging):
        n_rel = int(kg_df['r'].max()) + 1
        inv = kg_df.rename(columns={'h': 't', 't': 'h'}).copy()
        inv['r'] = inv['r'] + n_rel
        kg_df = pd.concat([kg_df, inv], axis=0, ignore_index=True)
        
        kg_df['r'] = kg_df['r'] + 2

        e_max = int(max(kg_df['h'].max(), kg_df['t'].max()))

        self.n_entities = max(self.n_items - 1, e_max) + 1
        self.n_relations = int(kg_df['r'].max()) + 1

        self.adj_entity, self.adj_relation = self._build_entity_adj(
            kg_df, self.n_entities, self.n_relations, self.neighbor_sample_size
        )  

        self.user_pos_items = {
            u: torch.as_tensor(sorted(list(items)), dtype=torch.long)
            for u, items in self.train_user_dict.items()
        }

        logging.info('n_users:           %d', self.n_users)
        logging.info('n_items:           %d', self.n_items)
        logging.info('n_entities:        %d', self.n_entities)
        logging.info('n_relations:       %d', self.n_relations)
        logging.info('neighbor_sample K: %d', self.neighbor_sample_size)
        logging.info('n_cf_train:        %d', self.n_cf_train)
        logging.info('n_cf_test:         %d', self.n_cf_test)

    @staticmethod
    def _build_entity_adj(kg_df: pd.DataFrame, n_entity: int, n_relation: int, K: int):
        neigh_e = [[] for _ in range(n_entity)]
        neigh_r = [[] for _ in range(n_entity)]

        for h, r, t in kg_df[['h', 'r', 't']].itertuples(index=False):
            if 0 <= h < n_entity and 0 <= t < n_entity and 0 <= r < n_relation:
                neigh_e[h].append(t)
                neigh_r[h].append(r)

        adj_e = np.zeros((n_entity, K), dtype=np.int64)
        adj_r = np.zeros((n_entity, K), dtype=np.int64)
        rng = np.random.default_rng()
        for e in range(n_entity):
            if not neigh_e[e]:
                adj_e[e, :] = e
                adj_r[e, :] = 0
                continue
            m = len(neigh_e[e])
            if m >= K:
                idx = rng.choice(m, size=K, replace=False)
                adj_e[e, :] = np.asarray([neigh_e[e][j] for j in idx], dtype=np.int64)
                adj_r[e, :] = np.asarray([neigh_r[e][j] for j in idx], dtype=np.int64)
            else:
                reps = (K + m - 1) // m
                ee = (neigh_e[e] * reps)[:K]
                rr = (neigh_r[e] * reps)[:K]
                adj_e[e, :] = np.asarray(ee, dtype=np.int64)
                adj_r[e, :] = np.asarray(rr, dtype=np.int64)
        return adj_e, adj_r
