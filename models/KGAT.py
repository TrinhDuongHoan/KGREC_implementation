# models/KGAT.py
import os
import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def _xavier_(t):
    if t is None: return
    if t.dim() >= 2:
        nn.init.xavier_uniform_(t)
    else:
        nn.init.zeros_(t)


def _to_torch_sparse(mx_coo):
    mx = mx_coo.tocoo().astype(np.float32)
    idx = torch.tensor(np.vstack([mx.row, mx.col]), dtype=torch.long)
    val = torch.tensor(mx.data, dtype=torch.float32)
    shape = torch.Size(mx.shape)
    return torch.sparse_coo_tensor(idx, val, shape).coalesce()


class KGAT(torch.nn.Module):
    """
    Bản PyTorch giữ nguyên tên/định nghĩa hàm giống TensorFlow:
      - __init__, _parse_args, _build_inputs, _build_weights,
        _build_model_phase_I, _build_loss_phase_I,
        _build_model_phase_II, _build_loss_phase_II,
        _statistics_params, train, train_A, eval, update_attentive_A,
        _create_bi_interaction_embed, _create_gcn_embed, _create_graphsage_embed,
        _split_A_hat, _convert_sp_mat_to_sp_tensor, _create_attentive_A_out, _generate_transE_score.
    Ghi chú:
      - sess không dùng. Chấp nhận đối số để không đổi chữ ký.
      - feed_dict là dict như bản TF: keys: users, pos_items, neg_items, h, r, pos_t, neg_t, node_dropout, mess_dropout.
    """

    def __init__(self, data_config, pretrain_data, args):
        super().__init__()
        self._parse_args(data_config, pretrain_data, args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._build_inputs()
        self.weights = self._build_weights()  # ParameterDict registered in module
        self._build_model_phase_I()
        self._build_loss_phase_I()
        self._build_model_phase_II()
        self._build_loss_phase_II()
        self._statistics_params()

    # ---------- args ----------
    def _parse_args(self, data_config, pretrain_data, args):
        self.model_type = 'kgat'
        self.pretrain_data = pretrain_data

        self.n_users      = data_config['n_users']
        self.n_items      = data_config['n_items']
        self.n_entities   = data_config['n_entities']
        self.n_relations  = data_config['n_relations']

        self.n_fold = 100

        # A_in và các list cạnh để update attention
        self.A_in        = data_config['A_in']           # scipy.sparse coo
        self.all_h_list  = np.asarray(data_config['all_h_list'], dtype=np.int64)
        self.all_r_list  = np.asarray(data_config['all_r_list'], dtype=np.int64)
        self.all_t_list  = np.asarray(data_config['all_t_list'], dtype=np.int64)
        self.all_v_list  = np.asarray(data_config['all_v_list'], dtype=np.float32)

        self.adj_uni_type = args.adj_uni_type
        self.lr            = args.lr

        self.emb_dim   = args.embed_size
        self.batch_size = args.batch_size

        self.kge_dim      = args.kge_size
        self.batch_size_kg = args.batch_size_kg

        self.weight_size = list(eval(args.layer_size))
        self.n_layers    = len(self.weight_size)

        self.alg_type = args.alg_type  # 'bi','kgat','gcn','graphsage'
        self.model_type += '_%s_%s_%s_l%d' % (args.adj_type, args.adj_uni_type, args.alg_type, self.n_layers)

        self.regs    = list(eval(args.regs))
        self.verbose = args.verbose

        # giữ đúng API: message dropout là list độ dài n_layers
        self.mess_dropout_cfg = [0.0] * self.n_layers

    # ---------- inputs place-holders (giữ tên) ----------
    def _build_inputs(self):
        # Không dùng placeholder. Lưu tên để giữ API feed_dict.
        self.users    = 'users'
        self.pos_items= 'pos_items'
        self.neg_items= 'neg_items'

        self.A_values = 'A_values'
        self.h        = 'h'
        self.r        = 'r'
        self.pos_t    = 'pos_t'
        self.neg_t    = 'neg_t'

        self.node_dropout = 'node_dropout'
        self.mess_dropout = 'mess_dropout'

    # ---------- params ----------
    def _build_weights(self):
        W = dict()

        # embeddings
        if self.pretrain_data is None:
            W['user_embed']   = nn.Parameter(torch.empty(self.n_users, self.emb_dim))
            W['entity_embed'] = nn.Parameter(torch.empty(self.n_entities, self.emb_dim))
            _xavier_(W['user_embed']); _xavier_(W['entity_embed'])
        else:
            W['user_embed'] = nn.Parameter(
                torch.tensor(self.pretrain_data['user_embed'], dtype=torch.float32))
            item_embed = torch.tensor(self.pretrain_data['item_embed'], dtype=torch.float32)
            other = torch.empty(self.n_entities - self.n_items, self.emb_dim); _xavier_(other)
            ent = torch.cat([item_embed, other], dim=0)
            W['entity_embed'] = nn.Parameter(ent)

        W['relation_embed'] = nn.Parameter(torch.empty(self.n_relations, self.kge_dim))
        W['trans_W']        = nn.Parameter(torch.empty(self.n_relations, self.emb_dim, self.kge_dim))
        _xavier_(W['relation_embed']); _xavier_(W['trans_W'])

        # layer sizes
        self.weight_size_list = [self.emb_dim] + self.weight_size

        # conv weights (giữ đúng tên)
        for k in range(self.n_layers):
            W['W_gc_%d' % k] = nn.Parameter(torch.empty(self.weight_size_list[k], self.weight_size_list[k+1]))
            W['b_gc_%d' % k] = nn.Parameter(torch.empty(1, self.weight_size_list[k+1]))

            W['W_bi_%d' % k] = nn.Parameter(torch.empty(self.weight_size_list[k], self.weight_size_list[k+1]))
            W['b_bi_%d' % k] = nn.Parameter(torch.empty(1, self.weight_size_list[k+1]))

            W['W_mlp_%d' % k] = nn.Parameter(torch.empty(2*self.weight_size_list[k], self.weight_size_list[k+1]))
            W['b_mlp_%d' % k] = nn.Parameter(torch.empty(1, self.weight_size_list[k+1]))

            _xavier_(W['W_gc_%d' % k]); _xavier_(W['b_gc_%d' % k])
            _xavier_(W['W_bi_%d' % k]); _xavier_(W['b_bi_%d' % k])
            _xavier_(W['W_mlp_%d' % k]); _xavier_(W['b_mlp_%d' % k])

        # move all parameters to target device once to avoid device mismatch during sparse mm
        for key, param in list(W.items()):
            if isinstance(param, nn.Parameter):
                W[key] = nn.Parameter(param.to(self.device))

        # optimizers (not registered buffers; recreated on init)
        self._opt  = torch.optim.Adam([p for p in W.values()], lr=self.lr)
        self._opt2 = torch.optim.Adam([p for p in W.values()], lr=self.lr)

        # sparse A
        self.A_torch = _to_torch_sparse(self.A_in).to(self.device)

        # cache tens
        self._all_user_entity = None  # concat(user, entity) weight view
        return nn.ParameterDict(W)

    # ---------- state dict helpers for saving/loading ----------
    def state_dict(self, *args, **kwargs):
        # expose weights only; other runtime tensors (A_torch etc.) are rebuilt on init
        sd = {}
        for k, v in self.weights.items():
            sd[f'weights.{k}'] = v.data
        return sd

    def load_state_dict(self, state_dict, strict=True):
        missing = []
        for k, v in state_dict.items():
            if k.startswith('weights.'):
                wk = k.split('weights.', 1)[1]
                if wk in self.weights:
                    self.weights[wk].data.copy_(v.to(self.weights[wk].data.device))
                else:
                    missing.append(wk)
        if strict and missing:
            raise KeyError(f'Missing weights: {missing}')
        return

    # ---------- model phase I ----------
    def _build_model_phase_I(self):
        # tạo các hook tên như TF; thực tính sẽ làm trong train/eval
        self.ua_embeddings = None
        self.ea_embeddings = None
        self.batch_predictions = None

    def _message_dropout(self, x, p):
        if p <= 0: return x
        return F.dropout(x, p=p, training=self._training_flag)

    def _create_bi_interaction_embed(self):
        A_fold_hat = self._split_A_hat(self.A_in)

        user = self.weights['user_embed']
        entity = self.weights['entity_embed']
        ego = torch.cat([user, entity], dim=0)
        all_emb = [ego]

        x = ego
        for k in range(self.n_layers):
            # sum messages
            temp = []
            for fA in A_fold_hat:
                temp.append(torch.sparse.mm(_to_torch_sparse(fA).to(self.device), x))
            side = torch.cat(temp, dim=0)

            add = x + side
            sum_emb = F.leaky_relu(add @ self.weights['W_gc_%d'%k] + self.weights['b_gc_%d'%k])

            bi = x * side
            bi_emb = F.leaky_relu(bi @ self.weights['W_bi_%d'%k] + self.weights['b_bi_%d'%k])

            x = sum_emb + bi_emb
            x = self._message_dropout(x, self.mess_dropout_cfg[k])
            x = F.normalize(x, p=2, dim=1)
            all_emb.append(x)

        all_emb = torch.cat(all_emb, dim=1)
        ua, ea = torch.split(all_emb, [self.n_users, self.n_entities], dim=0)
        return ua, ea

    def _create_gcn_embed(self):
        A_fold_hat = self._split_A_hat(self.A_in)
        x = torch.cat([self.weights['user_embed'], self.weights['entity_embed']], dim=0)
        all_emb = [x]
        for k in range(self.n_layers):
            temp = []
            for fA in A_fold_hat:
                temp.append(torch.sparse.mm(_to_torch_sparse(fA).to(self.device), x))
            x = torch.cat(temp, dim=0)
            x = F.leaky_relu(x @ self.weights['W_gc_%d'%k] + self.weights['b_gc_%d'%k])
            x = self._message_dropout(x, self.mess_dropout_cfg[k])
            x = F.normalize(x, p=2, dim=1)
            all_emb.append(x)
        all_emb = torch.cat(all_emb, dim=1)
        ua, ea = torch.split(all_emb, [self.n_users, self.n_entities], dim=0)
        return ua, ea

    def _create_graphsage_embed(self):
        A_fold_hat = self._split_A_hat(self.A_in)
        pre = torch.cat([self.weights['user_embed'], self.weights['entity_embed']], dim=0)
        all_emb = [pre]
        for k in range(self.n_layers):
            temp = []
            for fA in A_fold_hat:
                temp.append(torch.sparse.mm(_to_torch_sparse(fA).to(self.device), pre))
            neigh = torch.cat(temp, dim=0)
            cat = torch.cat([pre, neigh], dim=1)
            pre = F.relu(cat @ self.weights['W_mlp_%d'%k] + self.weights['b_mlp_%d'%k])
            pre = self._message_dropout(pre, self.mess_dropout_cfg[k])
            pre = F.normalize(pre, p=2, dim=1)
            all_emb.append(pre)
        all_emb = torch.cat(all_emb, dim=1)
        ua, ea = torch.split(all_emb, [self.n_users, self.n_entities], dim=0)
        return ua, ea


    def _build_loss_phase_I(self):
        self.base_loss = torch.tensor(0.0, device=self.device)
        self.kge_loss  = torch.tensor(0.0, device=self.device)
        self.reg_loss  = torch.tensor(0.0, device=self.device)
        self.loss      = torch.tensor(0.0, device=self.device)

    def _build_model_phase_II(self):
        # đặt placeholder biến tên như TF
        self.h_e = None; self.r_e = None; self.pos_t_e = None; self.neg_t_e = None
        self.A_kg_score = None
        self.A_out = None

    def _build_loss_phase_II(self):
        self.kge_loss2 = torch.tensor(0.0, device=self.device)
        self.reg_loss2 = torch.tensor(0.0, device=self.device)
        self.loss2     = torch.tensor(0.0, device=self.device)

    def _statistics_params(self):
        if self.verbose > 0:
            total = 0
            for p in self.weights.values():
                total += p.numel()
            print("#params: %d" % total)

    def _split_A_hat(self, X):
        A_fold_hat = []
        fold_len = (self.n_users + self.n_entities) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            end = self.n_users + self.n_entities if i_fold == self.n_fold - 1 else (i_fold + 1) * fold_len
            A_fold_hat.append(X.tocsr()[start:end, :].tocoo())
        return A_fold_hat

    def _convert_sp_mat_to_sp_tensor(self, X):
        return _to_torch_sparse(X)

    def _create_attentive_A_out(self, A_values):
        # softmax theo hàng trên sparse
        idx = torch.tensor(np.vstack([self.all_h_list, self.all_t_list]), dtype=torch.long, device=self.device)
        val = torch.tensor(A_values, dtype=torch.float32, device=self.device)
        shape = (self.n_users + self.n_entities, self.n_users + self.n_entities)
        A = torch.sparse_coo_tensor(idx, val, size=shape).coalesce()
        A = torch.sparse.softmax(A, dim=1).coalesce()
        return A

    def _generate_transE_score(self, h_idx, t_idx, r_idx):
        all_emb = torch.cat([self.weights['user_embed'], self.weights['entity_embed']], dim=0)
        h = all_emb[h_idx.to(self.device)]
        t = all_emb[t_idx.to(self.device)]
        r = self.weights['relation_embed'][r_idx.to(self.device)]
        W = self.weights['trans_W'][r_idx.to(self.device)]  # [B, d, k]
        h_k = torch.bmm(h.unsqueeze(1), W).squeeze(1)
        t_k = torch.bmm(t.unsqueeze(1), W).squeeze(1)
        score = torch.sum(t_k * torch.tanh(h_k + r), dim=1)
        return score

    # ---------- public API như TF ----------
    def train(self, sess, feed_dict):
        self._training_flag = True
        users    = torch.tensor(feed_dict[self.users],    dtype=torch.long, device=self.device)
        pos_items= torch.tensor(feed_dict[self.pos_items],dtype=torch.long, device=self.device)
        neg_items= torch.tensor(feed_dict[self.neg_items],dtype=torch.long, device=self.device)

        # phase I embeddings
        if self.alg_type in ['bi', 'kgat']:
            ua, ea = self._create_bi_interaction_embed()
        elif self.alg_type in ['gcn']:
            ua, ea = self._create_gcn_embed()
        elif self.alg_type in ['graphsage']:
            ua, ea = self._create_graphsage_embed()
        else:
            raise ValueError("alg_type invalid.")
        self.ua_embeddings, self.ea_embeddings = ua, ea

        # users in DataLoaderKGAT are offset by n_entities; map back to [0, n_users)
        u_idx = users
        if u_idx.numel() > 0 and int(torch.max(u_idx).item()) >= self.n_users:
            u_idx = u_idx - self.n_entities

        u_e   = ua[u_idx]
        pos_e = ea[pos_items]
        neg_e = ea[neg_items]

        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)

        base_loss = F.softplus(-(pos_scores - neg_scores)).mean()
        reg = (u_e.pow(2).sum() + pos_e.pow(2).sum() + neg_e.pow(2).sum()) / self.batch_size
        reg = self.regs[0] * reg

        self.base_loss = base_loss
        self.kge_loss  = torch.tensor(0.0, device=self.device)
        self.reg_loss  = reg
        self.loss      = self.base_loss + self.kge_loss + self.reg_loss

        self._opt.zero_grad()
        self.loss.backward()
        self._opt.step()

        # batch_predictions cho eval(users vs pos batch)
        self.batch_predictions = (u_e @ pos_e.t()).detach()
        return None, float(self.loss.item()), float(self.base_loss.item()), float(self.kge_loss.item()), float(self.reg_loss.item())

    def train_A(self, sess, feed_dict):
        self._training_flag = True
        h   = torch.tensor(feed_dict[self.h],   dtype=torch.long, device=self.device)
        r   = torch.tensor(feed_dict[self.r],   dtype=torch.long, device=self.device)
        pos = torch.tensor(feed_dict[self.pos_t], dtype=torch.long, device=self.device)
        neg = torch.tensor(feed_dict[self.neg_t], dtype=torch.long, device=self.device)

        # embeddings chuyển sang không cần l2 normalize bắt buộc như TF comment
        all_emb = torch.cat([self.weights['user_embed'], self.weights['entity_embed']], dim=0)
        h_e   = all_emb[h];    pos_t_e = all_emb[pos];   neg_t_e = all_emb[neg]
        r_e   = self.weights['relation_embed'][r]
        W     = self.weights['trans_W'][r]

        h_k   = torch.bmm(h_e.unsqueeze(1), W).squeeze(1)
        pos_k = torch.bmm(pos_t_e.unsqueeze(1), W).squeeze(1)
        neg_k = torch.bmm(neg_t_e.unsqueeze(1), W).squeeze(1)

        pos_kg = (h_k + r_e - pos_k).pow(2).sum(dim=1)
        neg_kg = (h_k + r_e - neg_k).pow(2).sum(dim=1)

        kg_loss = F.softplus(-(neg_kg - pos_kg)).mean()
        kg_reg  = (h_k.pow(2).sum() + r_e.pow(2).sum() + pos_k.pow(2).sum() + neg_k.pow(2).sum()) / max(1, h_k.size(0))
        kg_reg  = self.regs[1] * kg_reg

        self.kge_loss2 = kg_loss
        self.reg_loss2 = kg_reg
        self.loss2     = self.kge_loss2 + self.reg_loss2

        self._opt2.zero_grad()
        self.loss2.backward()
        self._opt2.step()

        return None, float(self.loss2.item()), float(self.kge_loss2.item()), float(self.reg_loss2.item())

    def eval(self, sess, feed_dict):
        self._training_flag = False
        users = torch.tensor(feed_dict[self.users], dtype=torch.long, device=self.device)
        pos   = torch.tensor(feed_dict[self.pos_items], dtype=torch.long, device=self.device)

        if self.alg_type in ['bi','kgat']:
            ua, ea = self._create_bi_interaction_embed()
        elif self.alg_type in ['gcn']:
            ua, ea = self._create_gcn_embed()
        else:
            ua, ea = self._create_graphsage_embed()

        u_idx = users
        if u_idx.numel() > 0 and int(torch.max(u_idx).item()) >= self.n_users:
            u_idx = u_idx - self.n_entities
        u = ua[u_idx]
        i = ea[pos]
        self.batch_predictions = (u @ i.t()).detach()
        return self.batch_predictions.cpu().numpy()

    def update_attentive_A(self, sess):
        # chẻ theo n_fold như TF để tiết kiệm RAM
        fold_len = len(self.all_h_list) // self.n_fold
        scores = []
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            end = len(self.all_h_list) if i_fold == self.n_fold - 1 else (i_fold + 1) * fold_len
            h = torch.tensor(self.all_h_list[start:end], dtype=torch.long)
            r = torch.tensor(self.all_r_list[start:end], dtype=torch.long)
            t = torch.tensor(self.all_t_list[start:end], dtype=torch.long)
            with torch.no_grad():
                sc = self._generate_transE_score(h, t, r).cpu().numpy()
            scores.append(sc)
        scores = np.concatenate(scores, axis=0).astype(np.float32)

        # softmax row-wise
        new_A = self._create_attentive_A_out(scores)
        self.A_torch = new_A.to(self.device)

        # cập nhật A_in scipy cho đồng bộ với API cũ
        nz = new_A.coalesce()
        rows = nz.indices()[0].cpu().numpy()
        cols = nz.indices()[1].cpu().numpy()
        vals = nz.values().cpu().numpy()
        self.A_in = sp.coo_matrix((vals, (rows, cols)),
                                  shape=(self.n_users + self.n_entities,
                                         self.n_users + self.n_entities))

        if self.alg_type in ['org', 'gcn']:
            self.A_in.setdiag(1.)

