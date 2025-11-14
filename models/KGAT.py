import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def _xavier_(t):
    if t is None:
        return
    if t.dim() >= 2:
        nn.init.xavier_uniform_(t)
    else:
        nn.init.zeros_(t)


def _to_torch_sparse(mx_coo: sp.coo_matrix) -> torch.Tensor:
    mx = mx_coo.tocoo().astype(np.float32)
    idx = torch.tensor(np.vstack([mx.row, mx.col]), dtype=torch.long)
    val = torch.tensor(mx.data, dtype=torch.float32)
    shape = torch.Size(mx.shape)
    return torch.sparse_coo_tensor(idx, val, shape).coalesce()


class KGAT(torch.nn.Module):

    def __init__(self, data_config, pretrain_data, args):
        super().__init__()
        self._parse_args(data_config, pretrain_data, args)
        self.device = getattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self._build_inputs()
        self.weights = self._build_weights()
        self._build_model_phase_I()
        self._build_loss_phase_I()
        self._build_model_phase_II()
        self._build_loss_phase_II()
        self._statistics_params()

        self._A_folds = None
        self._A_folds_dirty = True
        self._emb_cache = None
        self._emb_cache_dirty = True

    # ---------- args ----------
    def _parse_args(self, data_config, pretrain_data, args):
        self.model_type = 'kgat'
        self.pretrain_data = pretrain_data

        self.n_users     = data_config['n_users']
        self.n_items     = data_config['n_items']
        self.n_entities  = data_config['n_entities']
        self.n_relations = data_config['n_relations']

        self.n_fold = int(getattr(args, 'n_fold', 100))

        self.A_in       = data_config['A_in']          # scipy.sparse coo
        self.all_h_list = np.asarray(data_config['all_h_list'], dtype=np.int64)
        self.all_r_list = np.asarray(data_config['all_r_list'], dtype=np.int64)
        self.all_t_list = np.asarray(data_config['all_t_list'], dtype=np.int64)
        self.all_v_list = np.asarray(data_config['all_v_list'], dtype=np.float32)

        self.adj_uni_type = args.adj_uni_type
        self.lr           = args.lr

        self.emb_dim    = args.embed_size
        self.batch_size = args.batch_size

        self.kge_dim       = args.kge_size
        self.batch_size_kg = args.batch_size_kg

        self.weight_size = list(eval(args.layer_size))
        self.n_layers    = len(self.weight_size)

        self.alg_type = args.alg_type
        self.model_type += '_%s_%s_%s_l%d' % (args.adj_type, args.adj_uni_type, args.alg_type, self.n_layers)

        self.regs    = list(eval(args.regs))
        self.verbose = args.verbose

        # mess_dropout từ YAML (giống KGRec)
        if hasattr(args, "mess_dropout"):
            if isinstance(args.mess_dropout, str):
                self.mess_dropout_cfg = list(eval(args.mess_dropout))
            else:
                self.mess_dropout_cfg = list(args.mess_dropout)
        else:
            self.mess_dropout_cfg = [0.0] * self.n_layers

        if len(self.mess_dropout_cfg) != self.n_layers:
            raise ValueError(
                f"mess_dropout length ({len(self.mess_dropout_cfg)}) != n_layers ({self.n_layers})."
            )

    def _build_inputs(self):
        self.users     = 'users'
        self.pos_items = 'pos_items'
        self.neg_items = 'neg_items'

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
            _xavier_(W['user_embed'])
            _xavier_(W['entity_embed'])
        else:
            W['user_embed'] = nn.Parameter(
                torch.tensor(self.pretrain_data['user_embed'], dtype=torch.float32)
            )
            item_embed = torch.tensor(self.pretrain_data['item_embed'], dtype=torch.float32)
            other = torch.empty(self.n_entities - self.n_items, self.emb_dim)
            _xavier_(other)
            ent = torch.cat([item_embed, other], dim=0)
            W['entity_embed'] = nn.Parameter(ent)

        W['relation_embed'] = nn.Parameter(torch.empty(self.n_relations, self.kge_dim))
        W['trans_W']        = nn.Parameter(torch.empty(self.n_relations, self.emb_dim, self.kge_dim))
        _xavier_(W['relation_embed'])
        _xavier_(W['trans_W'])

        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            W[f'W_gc_{k}'] = nn.Parameter(torch.empty(self.weight_size_list[k], self.weight_size_list[k + 1]))
            W[f'b_gc_{k}'] = nn.Parameter(torch.empty(1, self.weight_size_list[k + 1]))

            W[f'W_bi_{k}'] = nn.Parameter(torch.empty(self.weight_size_list[k], self.weight_size_list[k + 1]))
            W[f'b_bi_{k}'] = nn.Parameter(torch.empty(1, self.weight_size_list[k + 1]))

            W[f'W_mlp_{k}'] = nn.Parameter(torch.empty(2 * self.weight_size_list[k], self.weight_size_list[k + 1]))
            W[f'b_mlp_{k}'] = nn.Parameter(torch.empty(1, self.weight_size_list[k + 1]))

            _xavier_(W[f'W_gc_{k}'])
            _xavier_(W[f'b_gc_{k}'])
            _xavier_(W[f'W_bi_{k}'])
            _xavier_(W[f'b_bi_{k}'])
            _xavier_(W[f'W_mlp_{k}'])
            _xavier_(W[f'b_mlp_{k}'])

        for key, param in list(W.items()):
            if isinstance(param, nn.Parameter):
                W[key] = nn.Parameter(param.to(self.device))

        self._opt  = torch.optim.Adam([p for p in W.values()], lr=self.lr)
        self._opt2 = torch.optim.Adam([p for p in W.values()], lr=self.lr)

        self.A_torch = _to_torch_sparse(self.A_in).to(self.device)

        self._all_user_entity = None
        return nn.ParameterDict(W)

    # ---------- state dict ----------
    def state_dict(self, *args, **kwargs):
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

    # ---------- phase I ----------
    def _build_model_phase_I(self):
        self.ua_embeddings = None
        self.ea_embeddings = None
        self.batch_predictions = None

    def _message_dropout(self, x, p):
        if p <= 0:
            return x
        return F.dropout(x, p=p, training=self._training_flag)

    def _create_bi_interaction_embed(self):
        A_fold_hat = self._get_A_folds()

        user = self.weights['user_embed']
        entity = self.weights['entity_embed']
        ego = torch.cat([user, entity], dim=0)
        all_emb = [ego]

        x = ego
        for k in range(self.n_layers):
            temp = []
            for fA in A_fold_hat:
                temp.append(torch.sparse.mm(fA, x))
            side = torch.cat(temp, dim=0)

            add = x + side
            sum_emb = F.leaky_relu(add @ self.weights[f'W_gc_{k}'] + self.weights[f'b_gc_{k}'])

            bi = x * side
            bi_emb = F.leaky_relu(bi @ self.weights[f'W_bi_{k}'] + self.weights[f'b_bi_{k}'])

            x = sum_emb + bi_emb
            x = self._message_dropout(x, self.mess_dropout_cfg[k])
            x = F.normalize(x, p=2, dim=1)
            all_emb.append(x)

        all_emb = torch.cat(all_emb, dim=1)
        ua, ea = torch.split(all_emb, [self.n_users, self.n_entities], dim=0)
        return ua, ea

    def _create_gcn_embed(self):
        A_fold_hat = self._get_A_folds()
        x = torch.cat([self.weights['user_embed'], self.weights['entity_embed']], dim=0)
        all_emb = [x]
        for k in range(self.n_layers):
            temp = []
            for fA in A_fold_hat:
                temp.append(torch.sparse.mm(fA, x))
            x = torch.cat(temp, dim=0)
            x = F.leaky_relu(x @ self.weights[f'W_gc_{k}'] + self.weights[f'b_gc_{k}'])
            x = self._message_dropout(x, self.mess_dropout_cfg[k])
            x = F.normalize(x, p=2, dim=1)
            all_emb.append(x)
        all_emb = torch.cat(all_emb, dim=1)
        ua, ea = torch.split(all_emb, [self.n_users, self.n_entities], dim=0)
        return ua, ea

    def _create_graphsage_embed(self):
        A_fold_hat = self._get_A_folds()
        pre = torch.cat([self.weights['user_embed'], self.weights['entity_embed']], dim=0)
        all_emb = [pre]
        for k in range(self.n_layers):
            temp = []
            for fA in A_fold_hat:
                temp.append(torch.sparse.mm(fA, pre))
            neigh = torch.cat(temp, dim=0)
            cat = torch.cat([pre, neigh], dim=1)
            pre = F.relu(cat @ self.weights[f'W_mlp_{k}'] + self.weights[f'b_mlp_{k}'])
            pre = self._message_dropout(pre, self.mess_dropout_cfg[k])
            pre = F.normalize(pre, p=2, dim=1)
            all_emb.append(pre)
        all_emb = torch.cat(all_emb, dim=1)
        ua, ea = torch.split(all_emb, [self.n_users, self.n_entities], dim=0)
        return ua, ea

    def _compute_embeddings_for_alg(self):
        if self.alg_type in ['bi', 'kgat']:
            return self._create_bi_interaction_embed()
        if self.alg_type in ['gcn']:
            return self._create_gcn_embed()
        if self.alg_type in ['graphsage']:
            return self._create_graphsage_embed()
        raise ValueError('alg_type invalid.')

    def get_embeddings(self):
        if self._emb_cache is None or self._emb_cache_dirty:
            ua, ea = self._compute_embeddings_for_alg()
            self._emb_cache = (ua.detach(), ea.detach())
            self._emb_cache_dirty = False
        ua, ea = self._emb_cache
        return ua, ea

    def _build_loss_phase_I(self):
        self.base_loss = torch.tensor(0.0, device=self.device)
        self.kge_loss  = torch.tensor(0.0, device=self.device)
        self.reg_loss  = torch.tensor(0.0, device=self.device)
        self.loss      = torch.tensor(0.0, device=self.device)

    def _build_model_phase_II(self):
        self.h_e = None
        self.r_e = None
        self.pos_t_e = None
        self.neg_t_e = None
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

    # ---------- utils ----------
    def _create_attentive_A_out(self, A_values: np.ndarray):
        idx = torch.tensor(
            np.vstack([self.all_h_list, self.all_t_list]),
            dtype=torch.long,
            device=self.device
        )
        val = torch.tensor(A_values, dtype=torch.float32, device=self.device)
        shape = (self.n_users + self.n_entities, self.n_users + self.n_entities)
        A = torch.sparse_coo_tensor(idx, val, size=shape).coalesce()
        A = torch.sparse.softmax(A, dim=1).coalesce()
        return A

    def _generate_transE_score(self, h_idx, t_idx, r_idx):
        all_emb = torch.cat([self.weights['user_embed'], self.weights['entity_embed']], dim=0)
        h = all_emb[h_idx]
        t = all_emb[t_idx]
        r = self.weights['relation_embed'][r_idx]
        W = self.weights['trans_W'][r_idx]  # [B, d, k]

        h_k = torch.bmm(h.unsqueeze(1), W).squeeze(1)
        t_k = torch.bmm(t.unsqueeze(1), W).squeeze(1)
        score = torch.sum(t_k * torch.tanh(h_k + r), dim=1)
        return score

    def _to_long_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=torch.long)
        return torch.as_tensor(x, device=self.device, dtype=torch.long)

    # ---------- train CF ----------
    def train(self, sess, feed_dict):
        self._training_flag = True

        users     = self._to_long_tensor(feed_dict[self.users])
        pos_items = self._to_long_tensor(feed_dict[self.pos_items])
        neg_items = self._to_long_tensor(feed_dict[self.neg_items])

        if self.alg_type in ['bi', 'kgat']:
            ua, ea = self._create_bi_interaction_embed()
        elif self.alg_type in ['gcn']:
            ua, ea = self._create_gcn_embed()
        elif self.alg_type in ['graphsage']:
            ua, ea = self._create_graphsage_embed()
        else:
            raise ValueError('alg_type invalid.')

        self.ua_embeddings, self.ea_embeddings = ua, ea

        u_idx = users
        if u_idx.numel() > 0 and int(torch.max(u_idx).item()) >= self.n_users:
            u_idx = u_idx - self.n_entities

        u_e   = ua[u_idx]
        pos_e = ea[pos_items]
        neg_e = ea[neg_items]

        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)

        base_loss = F.softplus(-(pos_scores - neg_scores)).mean()
        reg = (u_e.pow(2).sum() + pos_e.pow(2).sum() + neg_e.pow(2).sum()) / max(1, self.batch_size)
        reg = self.regs[0] * reg

        self.base_loss = base_loss
        self.kge_loss  = torch.tensor(0.0, device=self.device)
        self.reg_loss  = reg
        self.loss      = self.base_loss + self.kge_loss + self.reg_loss

        self._opt.zero_grad()
        self.loss.backward()
        self._opt.step()

        self.batch_predictions = (u_e @ pos_e.t()).detach()
        return (
            None,
            float(self.loss.item()),
            float(self.base_loss.item()),
            float(self.kge_loss.item()),
            float(self.reg_loss.item()),
        )

    # ---------- train KG ----------
    def train_A(self, sess, feed_dict):
        self._training_flag = True

        h   = self._to_long_tensor(feed_dict[self.h])
        r   = self._to_long_tensor(feed_dict[self.r])
        pos = self._to_long_tensor(feed_dict[self.pos_t])
        neg = self._to_long_tensor(feed_dict[self.neg_t])

        all_emb = torch.cat([self.weights['user_embed'], self.weights['entity_embed']], dim=0)
        h_e     = all_emb[h]
        pos_t_e = all_emb[pos]
        neg_t_e = all_emb[neg]
        r_e     = self.weights['relation_embed'][r]
        W       = self.weights['trans_W'][r]

        h_k   = torch.bmm(h_e.unsqueeze(1), W).squeeze(1)
        pos_k = torch.bmm(pos_t_e.unsqueeze(1), W).squeeze(1)
        neg_k = torch.bmm(neg_t_e.unsqueeze(1), W).squeeze(1)

        pos_kg = (h_k + r_e - pos_k).pow(2).sum(dim=1)
        neg_kg = (h_k + r_e - neg_k).pow(2).sum(dim=1)

        kg_loss = F.softplus(-(neg_kg - pos_kg)).mean()
        kg_reg  = (h_k.pow(2).sum() +
                   r_e.pow(2).sum() +
                   pos_k.pow(2).sum() +
                   neg_k.pow(2).sum()) / max(1, h_k.size(0))
        kg_reg  = self.regs[1] * kg_reg

        self.kge_loss2 = kg_loss
        self.reg_loss2 = kg_reg
        self.loss2     = self.kge_loss2 + self.reg_loss2

        self._opt2.zero_grad()
        self.loss2.backward()
        self._opt2.step()

        return (
            None,
            float(self.loss2.item()),
            float(self.kge_loss2.item()),
            float(self.reg_loss2.item()),
        )

    @torch.no_grad()
    def eval(self, sess, feed_dict):
        self._training_flag = False

        users = self._to_long_tensor(feed_dict[self.users])
        pos   = self._to_long_tensor(feed_dict[self.pos_items])

        ua, ea = self.get_embeddings()

        u_idx = users
        if u_idx.numel() > 0 and int(torch.max(u_idx).item()) >= self.n_users:
            u_idx = u_idx - self.n_entities

        u = ua[u_idx]
        i = ea[pos]
        self.batch_predictions = (u @ i.t()).detach()
        return self.batch_predictions.cpu().numpy()

    @torch.no_grad()
    def update_attentive_A(self, sess):
        fold_len = len(self.all_h_list) // self.n_fold
        scores = []
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            end = len(self.all_h_list) if i_fold == self.n_fold - 1 else (i_fold + 1) * fold_len

            h = torch.tensor(self.all_h_list[start:end], dtype=torch.long, device=self.device)
            r = torch.tensor(self.all_r_list[start:end], dtype=torch.long, device=self.device)
            t = torch.tensor(self.all_t_list[start:end], dtype=torch.long, device=self.device)

            sc = self._generate_transE_score(h, t, r).cpu().numpy()
            scores.append(sc)

        scores = np.concatenate(scores, axis=0).astype(np.float32)

        new_A = self._create_attentive_A_out(scores)
        self.A_torch = new_A.to(self.device)

        nz = new_A.coalesce()
        rows = nz.indices()[0].cpu().numpy()
        cols = nz.indices()[1].cpu().numpy()
        vals = nz.values().cpu().numpy()
        self.A_in = sp.coo_matrix(
            (vals, (rows, cols)),
            shape=(self.n_users + self.n_entities, self.n_users + self.n_entities)
        )

        if self.alg_type in ['org', 'gcn']:
            self.A_in.setdiag(1.0)

        self._emb_cache_dirty = True
        self._A_folds_dirty = True

    def _get_A_folds(self):
        if self._A_folds is None or self._A_folds_dirty:
            A_fold_hat = []
            fold_len = (self.n_users + self.n_entities) // self.n_fold
            for i_fold in range(self.n_fold):
                start = i_fold * fold_len
                end = self.n_users + self.n_entities if i_fold == self.n_fold - 1 else (i_fold + 1) * fold_len
                coo = self.A_in.tocsr()[start:end, :].tocoo()
                A_fold_hat.append(_to_torch_sparse(coo).to(self.device))
            self._A_folds = A_fold_hat
            self._A_folds_dirty = False
        return self._A_folds
