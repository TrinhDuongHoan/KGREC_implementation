import torch
import torch.nn as nn
import torch.nn.functional as F

class SumAggregator(nn.Module):
    def __init__(self, batch_size, dim, dropout=0.0, act=F.relu):
        super().__init__()
        self.batch_size = batch_size
        self.dim = dim
        self.act = act
        self.drop = nn.Dropout(p=dropout)
        self.lin = nn.Linear(dim, dim)

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        B, L, K, D = neighbor_vectors.shape
        u = user_embeddings.view(B, 1, 1, D)
        scores = (u * neighbor_relations).mean(dim=-1)           # [B,L,K]
        scores = F.softmax(scores, dim=-1).unsqueeze(-1)         # [B,L,K,1]
        agg = (scores * neighbor_vectors).mean(dim=2)            # [B,L,D]
        return agg

    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        B, L, D = self_vectors.shape
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)
        out = self_vectors + neighbors_agg
        out = self.drop(out.view(B * L, D))
        out = self.lin(out).view(B, L, D)
        return self.act(out)

class LabelAggregator(nn.Module):
    def __init__(self, batch_size, dim):
        super().__init__()
        self.batch_size = batch_size
        self.dim = dim

    def forward(self, self_labels, neighbor_labels, neighbor_relations, user_embeddings, masks):
        B, L, K = neighbor_labels.shape
        D = neighbor_relations.size(-1)
        u = user_embeddings.view(B, 1, 1, D)
        scores = (u * neighbor_relations).mean(dim=-1)           # [B,L,K]
        scores = F.softmax(scores, dim=-1)
        neigh = (scores * neighbor_labels).mean(dim=-1)          # [B,L]
        return masks.float() * self_labels + (~masks).float() * neigh

class KGNN_LS_Torch(nn.Module):
    def __init__(self, args, n_user, n_entity, n_relation,
                 adj_entity, adj_relation, user_pos_items):
        super().__init__()
        self.n_iter = args.n_iter
        self.batch_size = getattr(args, "test_batch_size", args.batch_size)
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.ls_weight = args.ls_weight

        self.register_buffer("adj_entity", torch.as_tensor(adj_entity, dtype=torch.long))
        self.register_buffer("adj_relation", torch.as_tensor(adj_relation, dtype=torch.long))
        self.register_buffer("_empty_pos", torch.empty(0, dtype=torch.long))
        self.user_pos_items = user_pos_items  # dict: u -> LongTensor(items), CPU

        self.user_emb = nn.Embedding(n_user, self.dim)
        self.entity_emb = nn.Embedding(n_entity, self.dim)
        self.relation_emb = nn.Embedding(n_relation, self.dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        self.sum_aggs = nn.ModuleList()
        for i in range(self.n_iter):
            act = torch.tanh if i == self.n_iter - 1 else F.relu
            self.sum_aggs.append(SumAggregator(self.batch_size, self.dim, dropout=0.0, act=act))
        self.label_agg = LabelAggregator(self.batch_size, self.dim)

        self.item_cache = None  # [n_items, D]
        self.bpr = lambda pos, neg: -torch.log(torch.sigmoid(pos - neg) + 1e-12).mean()

    @torch.no_grad()
    def _get_neighbors(self, seeds):
        seeds = seeds.view(-1, 1)
        entities = [seeds]
        relations = []
        for _ in range(self.n_iter):
            neigh_e = self.adj_entity[entities[-1]].view(seeds.size(0), -1)   # [B,K^hop]
            neigh_r = self.adj_relation[entities[-1]].view(seeds.size(0), -1) # [B,K^hop]
            entities.append(neigh_e)
            relations.append(neigh_r)
        return entities, relations

    def _aggregate(self, user_emb, entities, relations):
        entity_vectors = [self.entity_emb(e) for e in entities]      # [B,K^hop,D]
        relation_vectors = [self.relation_emb(r) for r in relations] # [B,K^hop,D]
        for i in range(self.n_iter):
            agg = self.sum_aggs[i]
            next_vecs = []
            for hop in range(self.n_iter - i):
                B = entity_vectors[hop].size(0)
                L = entity_vectors[hop].size(1)
                K = self.n_neighbor
                D = self.dim
                self_vec = entity_vectors[hop]                         # [B,L,D]
                neigh_vec = entity_vectors[hop + 1].view(B, L, K, D)   # [B,L,K,D]
                neigh_rel = relation_vectors[hop].view(B, L, K, D)     # [B,L,K,D]
                v = agg(self_vec, neigh_vec, neigh_rel, user_emb)      # [B,L,D]
                next_vecs.append(v)
            entity_vectors = next_vecs
        return entity_vectors[0].view(user_emb.size(0), self.dim)     # [B,D]

    def _label_smoothness(self, user_ids, item_ids):
        entities, relations = self._get_neighbors(item_ids)           # e[0]: [B,1], e[i]: [B,K^i]
        B = user_ids.size(0)
        holdout = entities[0]                                         # [B,1]

        entity_labels, reset_masks = [], []
        for e in entities:
            init_rows, reset_rows = [], []
            for b in range(B):
                u = int(user_ids[b].item())
                pos = self.user_pos_items.get(u, self._empty_pos).to(e.device)
                row = torch.isin(e[b], pos)                           # [K^i] bool
                init_rows.append(row.float())
                reset_rows.append(row)
            initial = torch.stack(init_rows, dim=0)                   # [B,K^i]
            reset = torch.stack(reset_rows, dim=0)                    # [B,K^i]
            holdout_mask = (holdout != e).squeeze(1)
            if holdout_mask.dim() == 1:
                holdout_mask = holdout_mask.unsqueeze(1)
            initial = holdout_mask.float() * initial + (~holdout_mask).float() * 0.5
            reset = reset & holdout_mask
            entity_labels.append(initial)
            reset_masks.append(reset)
        reset_masks = reset_masks[:-1]

        relation_vecs = [self.relation_emb(r) for r in relations]
        for i in range(self.n_iter):
            next_labels = []
            for hop in range(self.n_iter - i):
                B = entity_labels[hop].size(0)
                K = self.n_neighbor
                D = self.dim
                next_len = entity_labels[hop + 1].size(1)
                L = next_len // K
                self_lab = entity_labels[hop].view(B, L)              # [B,L]
                neigh_lab = entity_labels[hop + 1].view(B, L, K)      # [B,L,K]
                neigh_rel = relation_vecs[hop].view(B, L, K, D)       # [B,L,K,D]
                mask = reset_masks[hop].view(B, L)                    # [B,L]
                v = self.label_agg(self_lab, neigh_lab, neigh_rel, self.user_emb(user_ids), mask)  # [B,L]
                next_labels.append(v)
            entity_labels = next_labels

        pred_labels = entity_labels[0]
        if pred_labels.dim() > 1:
            pred_labels = pred_labels.squeeze(-1)
        return pred_labels.clamp(0.0, 1.0)

    def _score_pair(self, user_ids, item_ids):
        u = self.user_emb(user_ids)                                   # [B,D]
        e, r = self._get_neighbors(item_ids)
        i = self._aggregate(u, e, r)                                  # [B,D]
        return (u * i).sum(dim=1)                                     # [B]

    # --------- Caching for evaluation ----------
    @torch.no_grad()
    def build_item_cache(self, device, n_items=None, chunk=4096):
        self.item_cache = None
        I = int(n_items) if n_items is not None else self.entity_emb.num_embeddings
        item_ids = torch.arange(I, device=device, dtype=torch.long)
        # dùng u_fake để đi qua aggregator
        u_fake_ref = self.user_emb.weight.mean(dim=0, keepdim=True)
        cache = []
        for ch in item_ids.split(chunk):
            u_fake = u_fake_ref.expand(ch.size(0), -1)                # [chunk,D]
            e, r = self._get_neighbors(ch)
            i_emb = self._aggregate(u_fake, e, r)                     # [chunk,D]
            cache.append(i_emb)
        self.item_cache = torch.cat(cache, dim=0)                     # [I,D]

    @torch.no_grad()
    def predict_from_cache(self, user_ids):
        assert self.item_cache is not None, "Call build_item_cache() first."
        u = self.user_emb(user_ids)                                   # [U,D]
        return u @ self.item_cache.t()                                # [U,I]

    # -------------- Forward --------------
    def forward(self, a, b, mode='predict', **kwargs):
        if mode == 'predict':
            if self.item_cache is not None and kwargs.get("use_cache", True):
                return self.predict_from_cache(a)
            user_ids, item_ids = a, b
            U, I = user_ids.size(0), item_ids.size(0)
            u = self.user_emb(user_ids)                               # [U,D]
            u_fake = u.mean(dim=0, keepdim=True).expand(I, -1)        # [I,D]
            i_list = []
            for chunk in torch.split(item_ids, 512):
                e, r = self._get_neighbors(chunk)
                i_emb = self._aggregate(u_fake[:chunk.size(0)], e, r) # [chunk,D]
                i_list.append(i_emb)
            item_emb = torch.cat(i_list, dim=0)                       # [I,D]
            return u @ item_emb.t()                                   # [U,I]

        if mode == 'train_cf':
            user_ids = a
            pos_items = b
            neg_items = kwargs['neg_items']
            ls_weight_eff = kwargs.get('ls_weight_eff', self.ls_weight)
            ls_subsample = kwargs.get('ls_subsample', 256)

            pos = self._score_pair(user_ids, pos_items)
            neg = self._score_pair(user_ids, neg_items)
            bpr_loss = self.bpr(pos, neg)

            # LS on subsample
            M = min(ls_subsample, user_ids.size(0))
            idx = torch.randint(0, user_ids.size(0), (M,), device=user_ids.device)
            ls_pos = self._label_smoothness(user_ids[idx], pos_items[idx])
            ls_neg = self._label_smoothness(user_ids[idx], neg_items[idx])
            ls_loss = F.binary_cross_entropy(ls_pos.clamp(1e-6, 1-1e-6), torch.ones_like(ls_pos)) + \
                      F.binary_cross_entropy(ls_neg.clamp(1e-6, 1-1e-6), torch.zeros_like(ls_neg))

            l2 = (self.user_emb.weight.norm(2)**2 +
                  self.entity_emb.weight.norm(2)**2 +
                  self.relation_emb.weight.norm(2)**2) / self.dim

            return bpr_loss + self.l2_weight * l2 + ls_weight_eff * ls_loss

        if mode == 'train_kg':
            return torch.zeros((), device=self.user_emb.weight.device, dtype=torch.float32)

        if mode == 'update_att':
            return None

        raise ValueError(f"Unknown mode: {mode}")
