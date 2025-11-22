import torch
import torch.nn as nn
import torch.nn.functional as F

class CKANCore(nn.Module):
    def __init__(self, n_entity, n_relation, dim, n_layer, agg):
        super().__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = dim
        self.n_layer = n_layer
        self.agg = agg

        self.entity_emb = nn.Embedding(n_entity, dim)
        self.relation_emb = nn.Embedding(n_relation, dim)

        self.attention = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, 1, bias=False),
            nn.Sigmoid(),
        )
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    
    def _agg_embeddings(self, items, user_triple_set, item_triple_set):
        user_embeddings = []
        user_emb_0 = self.entity_emb(user_triple_set[0][0])      # [B, T, D]
        user_embeddings.append(user_emb_0.mean(dim=1))           # [B, D]
        for i in range(self.n_layer):
            h = self.entity_emb(user_triple_set[0][i])           # [B, T, D]
            r = self.relation_emb(user_triple_set[1][i])         # [B, T, D]
            t = self.entity_emb(user_triple_set[2][i])           # [B, T, D]
            user_embeddings.append(self._knowledge_attention(h, r, t))  # [B, D]

        item_embeddings = []
        item_origin = self.entity_emb(items)                     # [B, D]
        item_embeddings.append(item_origin)
        for i in range(self.n_layer):
            h = self.entity_emb(item_triple_set[0][i])
            r = self.relation_emb(item_triple_set[1][i])
            t = self.entity_emb(item_triple_set[2][i])
            item_embeddings.append(self._knowledge_attention(h, r, t))
        if self.n_layer > 0 and (self.agg in ('sum', 'pool')):
            item_emb0 = self.entity_emb(item_triple_set[0][0])   # [B, T, D]
            item_embeddings.append(item_emb0.mean(dim=1))        # [B, D]

        e_u = user_embeddings[0]
        e_v = item_embeddings[0]
        if self.agg == 'concat':
            assert len(user_embeddings) == len(item_embeddings)
            for i in range(1, len(user_embeddings)):
                e_u = torch.cat([user_embeddings[i], e_u], dim=-1)
                e_v = torch.cat([item_embeddings[i], e_v], dim=-1)
        elif self.agg == 'sum':
            for i in range(1, len(user_embeddings)):  e_u = e_u + user_embeddings[i]
            for i in range(1, len(item_embeddings)):  e_v = e_v + item_embeddings[i]
        elif self.agg == 'pool':
            for i in range(1, len(user_embeddings)):  e_u = torch.max(e_u, user_embeddings[i])
            for i in range(1, len(item_embeddings)):  e_v = torch.max(e_v, item_embeddings[i])
        else:
            raise ValueError(f"Unknown aggregator: {self.agg}")
        return e_u, e_v

    def _knowledge_attention(self, h_emb, r_emb, t_emb):
        att = self.attention(torch.cat([h_emb, r_emb], dim=-1)).squeeze(-1)  # [B, T]
        att = F.softmax(att, dim=-1)
        out = (att.unsqueeze(-1) * t_emb).sum(dim=1)  # [B, D]
        return out

    def logits(self, items, user_triple_set, item_triple_set):
        e_u, e_v = self._agg_embeddings(items, user_triple_set, item_triple_set)  # [B,D],[B,D or kD]
        return (e_u * e_v).sum(dim=1)  # [B]

    def scores(self, items, user_triple_set, item_triple_set):
        return torch.sigmoid(self.logits(items, user_triple_set, item_triple_set))


class CKAN(nn.Module):

    def __init__(self, n_users, n_entity, n_relation, dim=64, n_layer=1, agg='sum',
                 l2=1e-5):
        super().__init__()
        self.n_users = n_users
        self.core = CKANCore(n_entity, n_relation, dim, n_layer, agg)
        self.l2 = float(l2)

    def _l2(self):
        reg = torch.zeros((), device=next(self.parameters()).device)
        for p in self.parameters():
            if p.requires_grad:
                reg = reg + p.pow(2).sum()
        return reg

    def predict(self, users, items, user_triple_set, item_triple_set):
        return self.core.scores(items, user_triple_set, item_triple_set)  # [B]

    def bpr_loss(self, pos_logits, neg_logits):
        return -F.logsigmoid(pos_logits - neg_logits).mean()

    def forward(self, *inputs, mode="predict"):
        if mode == "predict":
            users, items, user_triple_set, item_triple_set = inputs
            return self.predict(users, items, user_triple_set, item_triple_set)

        if mode == "train_cf":
            (users, pos_items, neg_items,
             u_ts_pos, i_ts_pos, u_ts_neg, i_ts_neg) = inputs
            pos_logits = self.core.logits(pos_items, u_ts_pos, i_ts_pos)
            neg_logits = self.core.logits(neg_items, u_ts_neg, i_ts_neg)
            loss = self.bpr_loss(pos_logits, neg_logits) + self.l2 * self._l2()
            return loss

        raise ValueError("Unsupported mode")
