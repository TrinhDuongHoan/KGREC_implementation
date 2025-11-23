import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Sequence


class MCRec(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        path_nums: Sequence[int],
        timestamps: Sequence[int],
        feat_len: int,
        latent_dim: int = 64,
        conv_out_channels: int = 128,
        conv_kernel_size: int = 4,
        mlp_layers: Sequence[int] = (512, 256, 128, 64),
        dropout: float = 0.5,
        user_pre_embed: torch.Tensor | None = None,
        item_pre_embed: torch.Tensor | None = None,
    ):
        super().__init__()
        assert len(path_nums) == 4 and len(timestamps) == 4, "expected 4 metapath groups"

        self.n_users = n_users
        self.n_items = n_items
        self.path_nums = list(path_nums)
        self.timestamps = list(timestamps)
        self.feat_len = feat_len
        self.latent_dim = latent_dim

        self.user_emb = nn.Embedding(n_users, latent_dim)
        self.item_emb = nn.Embedding(n_items, latent_dim)
        if user_pre_embed is not None and user_pre_embed.shape[0] == n_users and user_pre_embed.shape[1] == latent_dim:
            self.user_emb.weight = nn.Parameter(user_pre_embed.clone())
        if item_pre_embed is not None and item_pre_embed.shape[0] == n_items and item_pre_embed.shape[1] == latent_dim:
            self.item_emb.weight = nn.Parameter(item_pre_embed.clone())

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(in_channels=feat_len, out_channels=conv_out_channels, kernel_size=conv_kernel_size)
                for _ in range(4)
            ]
        )

        self.dropout = nn.Dropout(dropout)

        att_input_size = latent_dim + latent_dim + conv_out_channels
        self.metapath_att_fc1 = nn.Linear(att_input_size, conv_out_channels)
        self.metapath_att_fc2 = nn.Linear(conv_out_channels, 1)

        self.user_att = nn.Linear(latent_dim + conv_out_channels, latent_dim)
        self.item_att = nn.Linear(latent_dim + conv_out_channels, latent_dim)

        mlp_input = latent_dim + conv_out_channels + latent_dim
        mlp_layers_full = [mlp_input] + list(mlp_layers)
        mlp_modules = []
        for i in range(len(mlp_layers_full) - 1):
            mlp_modules.append(nn.Linear(mlp_layers_full[i], mlp_layers_full[i + 1]))
            mlp_modules.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_modules)
        self.pred = nn.Linear(mlp_layers_full[-1], 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        for conv in self.conv_layers:
            nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu')
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0.0)
        nn.init.xavier_uniform_(self.metapath_att_fc1.weight)
        nn.init.constant_(self.metapath_att_fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.metapath_att_fc2.weight)
        nn.init.constant_(self.metapath_att_fc2.bias, 0.0)
        nn.init.xavier_uniform_(self.user_att.weight)
        nn.init.constant_(self.user_att.bias, 0.0)
        nn.init.xavier_uniform_(self.item_att.weight)
        nn.init.constant_(self.item_att.bias, 0.0)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        nn.init.xavier_uniform_(self.pred.weight)
        if self.pred.bias is not None:
            nn.init.constant_(self.pred.bias, 0.0)

    def _encode_path_group(self, x, conv):
        batch, path_num, timestamps, feat_len = x.shape
        x = x.view(batch * path_num, timestamps, feat_len).permute(0, 2, 1).contiguous()
        h = F.relu(conv(x))
        h, _ = torch.max(h, dim=2)
        h = self.dropout(h)
        h = h.view(batch, path_num, -1)
        return h

    def _metapath_attention(self, user_latent, item_latent, metapaths):
        batch, n_mp, d = metapaths.shape
        u = user_latent.unsqueeze(1).expand(-1, n_mp, -1)
        v = item_latent.unsqueeze(1).expand(-1, n_mp, -1)
        concat = torch.cat([u, v, metapaths], dim=2)
        scores = F.relu(self.metapath_att_fc1(concat))
        scores = self.metapath_att_fc2(scores).squeeze(-1)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        out = torch.sum(metapaths * weights, dim=1)
        return out

    def _dim_attention(self, latent, path_vec, att_layer):
        concat = torch.cat([latent, path_vec], dim=1)
        scores = F.relu(att_layer(concat))
        weights = F.softmax(scores, dim=1)
        return latent * weights

    def forward(self, user, item, umtm, umum, umtmum, uuum, mode: str = 'train'):
        user_latent = self.user_emb(user)
        item_latent = self.item_emb(item)

        umtm_h = self._encode_path_group(umtm, self.conv_layers[0])
        umum_h = self._encode_path_group(umum, self.conv_layers[1])
        umtmum_h = self._encode_path_group(umtmum, self.conv_layers[2])
        uuum_h = self._encode_path_group(uuum, self.conv_layers[3])

        def agg_group(h):
            return torch.max(h, dim=1)[0]

        g0 = agg_group(umtm_h)
        g1 = agg_group(umum_h)
        g2 = agg_group(umtmum_h)
        g3 = agg_group(uuum_h)

        stacked = torch.stack([g0, g1, g2, g3], dim=1)

        path_output = self._metapath_attention(user_latent, item_latent, stacked)

        user_atten = self._dim_attention(user_latent, path_output, self.user_att)
        item_atten = self._dim_attention(item_latent, path_output, self.item_att)

        x = torch.cat([user_atten, path_output, item_atten], dim=1)
        x = self.mlp(x)
        logits = self.pred(x)
        if mode == 'predict':
            return logits
        return torch.sigmoid(logits)


if __name__ == '__main__':
    batch = 2
    n_users, n_items = 100, 200
    path_nums = [3, 2, 2, 1]
    timestamps = [10, 8, 6, 5]
    feat_len = 64
    model = MCRec(n_users, n_items, path_nums, timestamps, feat_len)
    user = torch.randint(0, n_users, (batch,))
    item = torch.randint(0, n_items, (batch,))
    umtm = torch.randn(batch, path_nums[0], timestamps[0], feat_len)
    umum = torch.randn(batch, path_nums[1], timestamps[1], feat_len)
    umtmum = torch.randn(batch, path_nums[2], timestamps[2], feat_len)
    uuum = torch.randn(batch, path_nums[3], timestamps[3], feat_len)
    out = model(user, item, umtm, umum, umtmum, uuum)
    print('MCRec forward output shape:', out.shape)
