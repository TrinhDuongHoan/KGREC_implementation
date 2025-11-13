import os
import collections
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp

from loaders.base_loader import DataLoaderBase


class DataLoaderKGAT(DataLoaderBase):
    """
    Loader chuyên cho KGAT, giữ API tương thích với training/train_kgat.py.

    Thuộc tính được sử dụng bởi trainer/model:
    - n_users, n_items, n_entities, n_relations
    - n_users_entities
    - n_cf_train, n_cf_test, n_kg_train
    - train_user_dict, test_user_dict
    - train_kg_dict, train_relation_dict
    - h_list, t_list, r_list (torch.LongTensor)
    - A_in (torch.sparse.FloatTensor) — tổng Laplacian (random-walk) qua các quan hệ
    - batch_size: dùng cho CF phase (trainer truy cập data.batch_size)
    - test_batch_size: batch size người dùng cho evaluate
    """

    def __init__(self, args, logging):
        # Đảm bảo tương thích tên tham số embedding giữa loader base và KGAT
        if not hasattr(args, 'embed_dim') and hasattr(args, 'embed_size'):
            args.embed_dim = args.embed_size

        super().__init__(args, logging)  # load CF train/test; có thể load pretrain nếu use_pretrain==1

        # Thiết lập batch size tương thích KGAT trainer
        # data.batch_size sẽ được trainer dùng cho CF
        if hasattr(args, 'batch_size'):
            self.batch_size = args.batch_size
        elif hasattr(args, 'cf_batch_size'):
            self.batch_size = args.cf_batch_size
        else:
            self.batch_size = 1024

        # evaluate batch size cho users
        self.test_batch_size = getattr(args, 'test_batch_size', 512)

        # Tải KG, xây đồ thị và các cấu trúc cần thiết
        kg_data = self.load_kg(self.kg_file)
        self._construct_data_for_kgat(kg_data)
        self._build_relation_adjacency()
        self._build_laplacian_and_Ain()
        self._print_info(logging)

    # -------------------- helpers --------------------
    def _construct_data_for_kgat(self, kg_data: pd.DataFrame):
        """Mở rộng KG với cạnh ngược + ánh xạ CF vào không gian KG (users offset bằng n_entities)."""
        # thêm inverse edges cho KG
        n_relations = int(kg_data['r'].max()) + 1
        inv_kg = kg_data.rename({'h': 't', 't': 'h'}, axis='columns').copy()
        inv_kg['r'] = inv_kg['r'] + n_relations
        kg_data = pd.concat([kg_data, inv_kg], axis=0, ignore_index=True)

        # offset 2 như kgrec_loader (dành cho quan hệ đặc biệt từ CF)
        kg_data['r'] = kg_data['r'] + 2

        self.n_relations = int(kg_data['r'].max()) + 1
        self.n_entities = int(max(kg_data['h'].max(), kg_data['t'].max())) + 1
        self.n_users_entities = self.n_users + self.n_entities

        # Ánh xạ CF users sang không gian KG (offset n_entities)
        self.cf_train_data = (
            np.array([u + self.n_entities for u in self.cf_train_data[0]], dtype=np.int32),
            self.cf_train_data[1].astype(np.int32)
        )
        self.cf_test_data = (
            np.array([u + self.n_entities for u in self.cf_test_data[0]], dtype=np.int32),
            self.cf_test_data[1].astype(np.int32)
        )

        self.train_user_dict = {int(k) + self.n_entities: np.unique(v).astype(np.int32)
                                 for k, v in self.train_user_dict.items()}
        self.test_user_dict = {int(k) + self.n_entities: np.unique(v).astype(np.int32)
                                for k, v in self.test_user_dict.items()}

        # Thêm cạnh từ CF vào KG (cả chiều thuận/ngược)
        cf_edges = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf_edges['h'] = self.cf_train_data[0]
        cf_edges['t'] = self.cf_train_data[1]

        inv_cf_edges = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        inv_cf_edges['h'] = self.cf_train_data[1]
        inv_cf_edges['t'] = self.cf_train_data[0]

        self.kg_train_data = pd.concat([kg_data, cf_edges, inv_cf_edges], ignore_index=True)
        self.n_kg_train = len(self.kg_train_data)

        # Build dicts và lists
        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)
        h_list, t_list, r_list = [], [], []

        for _, row in self.kg_train_data.iterrows():
            h, r, t = int(row['h']), int(row['r']), int(row['t'])
            h_list.append(h); t_list.append(t); r_list.append(r)
            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))

        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)

    def _build_relation_adjacency(self):
        """Tạo adjacency per-relation (COO) kích thước [n_users_entities, n_users_entities]."""
        self.adjacency_dict = {}
        N = self.n_users_entities
        for r, ht in self.train_relation_dict.items():
            if len(ht) == 0:
                self.adjacency_dict[r] = sp.coo_matrix((N, N), dtype=np.float32)
                continue
            rows = np.fromiter((h for h, _ in ht), dtype=np.int64)
            cols = np.fromiter((t for _, t in ht), dtype=np.int64)
            vals = np.ones_like(rows, dtype=np.float32)
            self.adjacency_dict[r] = sp.coo_matrix((vals, (rows, cols)), shape=(N, N))

    @staticmethod
    def _symmetric_norm(adj: sp.coo_matrix) -> sp.coo_matrix:
        rowsum = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D = sp.diags(d_inv_sqrt)
        return (D @ adj @ D).tocoo()

    @staticmethod
    def _random_walk_norm(adj: sp.coo_matrix) -> sp.coo_matrix:
        rowsum = np.array(adj.sum(axis=1)).flatten()
        d_inv = np.power(rowsum, -1.0)
        d_inv[np.isinf(d_inv)] = 0.0
        D = sp.diags(d_inv)
        return (D @ adj).tocoo()

    def _build_laplacian_and_Ain(self, norm_type: str = 'random-walk'):
        norm_fn = self._random_walk_norm if norm_type == 'random-walk' else self._symmetric_norm
        self.laplacian_dict = {r: norm_fn(adj) for r, adj in self.adjacency_dict.items()}
        # Tổng tất cả quan hệ
        if len(self.laplacian_dict) == 0:
            A = sp.coo_matrix((self.n_users_entities, self.n_users_entities), dtype=np.float32)
        else:
            A = None
            for coo in self.laplacian_dict.values():
                A = coo if A is None else (A + coo)
        self.A_in = self._coo_to_torch_sparse(A.tocoo()) if A is not None else self._coo_to_torch_sparse(
            sp.coo_matrix((self.n_users_entities, self.n_users_entities), dtype=np.float32)
        )

    @staticmethod
    def _coo_to_torch_sparse(coo: sp.coo_matrix) -> torch.Tensor:
        i = torch.LongTensor(np.vstack([coo.row, coo.col]))
        v = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(i, v, torch.Size(coo.shape)).coalesce()

    def _print_info(self, logging):
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
