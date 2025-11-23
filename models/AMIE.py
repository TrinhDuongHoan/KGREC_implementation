from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AMIE(nn.Module):
	def __init__(
		self,
		args,
		n_users: int,
		n_items: int,
		user_pre_embed: torch.Tensor | None = None,
		item_pre_embed: torch.Tensor | None = None,
	) -> None:
		super().__init__()
		self.n_users = n_users
		self.n_items = n_items

		self.embed_dim = int(getattr(args, "embed_dim", 64))
		self.n_interest = int(getattr(args, "n_interest", 4))
		self.interest_dim = int(getattr(args, "interest_dim", self.embed_dim))
		self.temperature = float(max(getattr(args, "interest_temperature", 1.0), 1e-4))
		self.dropout_rate = float(getattr(args, "interest_dropout", 0.0))
		self.cf_l2loss_lambda = float(getattr(args, "cf_l2loss_lambda", 1e-4))
		self.use_pretrain = int(getattr(args, "use_pretrain", 0))
		self.predict_chunk_size = int(getattr(args, "predict_chunk_size", 2048))

		self.user_embedding = nn.Embedding(n_users, self.embed_dim)
		self.item_embedding = nn.Embedding(n_items, self.interest_dim)
		self.interest_proj = nn.Linear(self.embed_dim, self.n_interest * self.interest_dim)
		self.interest_gate = nn.Linear(self.embed_dim, self.n_interest)
		self.dropout = nn.Dropout(self.dropout_rate)

		self._init_parameters(user_pre_embed, item_pre_embed)

	def _init_parameters(
		self,
		user_pre_embed: torch.Tensor | None,
		item_pre_embed: torch.Tensor | None,
	) -> None:
		if user_pre_embed is not None:
			user_pre_embed = torch.as_tensor(user_pre_embed, dtype=torch.float32)
		if self.use_pretrain == 1 and user_pre_embed is not None and user_pre_embed.shape == (self.n_users, self.embed_dim):
			self.user_embedding.weight = nn.Parameter(user_pre_embed.clone().detach())
		else:
			nn.init.xavier_uniform_(self.user_embedding.weight)

		if item_pre_embed is not None:
			item_pre_embed = torch.as_tensor(item_pre_embed, dtype=torch.float32)
		if self.use_pretrain == 1 and item_pre_embed is not None and item_pre_embed.shape == (self.n_items, self.interest_dim):
			self.item_embedding.weight = nn.Parameter(item_pre_embed.clone().detach())
		else:
			nn.init.xavier_uniform_(self.item_embedding.weight)

		nn.init.xavier_uniform_(self.interest_proj.weight)
		nn.init.constant_(self.interest_proj.bias, 0.0)
		nn.init.xavier_uniform_(self.interest_gate.weight)
		nn.init.constant_(self.interest_gate.bias, 0.0)

	def _user_interests(self, user_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Return base user embedding, projected interests, and prior weights."""

		user_embed = self.user_embedding(user_ids)
		projected = self.interest_proj(user_embed).view(-1, self.n_interest, self.interest_dim)
		projected = self.dropout(projected)
		prior = torch.softmax(self.interest_gate(user_embed), dim=1)
		return user_embed, projected, prior

	def _score_items(
		self,
		interest_vectors: torch.Tensor,
		prior: torch.Tensor,
		item_embed: torch.Tensor,
	) -> torch.Tensor:
		"""Compute logits for a batch of items given interest vectors."""

		logits = torch.sum(interest_vectors * item_embed.unsqueeze(1), dim=2) / self.temperature
		logits = logits + torch.log(prior.clamp_min(1e-8))
		weights = torch.softmax(logits, dim=1)
		mixed_user = torch.sum(weights.unsqueeze(-1) * interest_vectors, dim=1)
		return torch.sum(mixed_user * item_embed, dim=1)

	def calc_cf_loss(
		self,
		user_ids: torch.Tensor,
		pos_item_ids: torch.Tensor,
		neg_item_ids: torch.Tensor,
	) -> torch.Tensor:
		user_embed, interests, prior = self._user_interests(user_ids)

		pos_embed = self.item_embedding(pos_item_ids)
		neg_embed = self.item_embedding(neg_item_ids)

		pos_logits = self._score_items(interests, prior, pos_embed)
		neg_logits = self._score_items(interests, prior, neg_embed)

		cf_loss = -F.logsigmoid(pos_logits - neg_logits).mean()
		if self.cf_l2loss_lambda > 0:
			l2 = (
				user_embed.pow(2).sum(dim=1)
				+ pos_embed.pow(2).sum(dim=1)
				+ neg_embed.pow(2).sum(dim=1)
			).mean()
			cf_loss = cf_loss + self.cf_l2loss_lambda * l2
		return cf_loss

	def calc_score(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
		user_embed, interests, prior = self._user_interests(user_ids)
		item_embed = self.item_embedding(item_ids)
		prior_log = torch.log(prior.clamp_min(1e-8)).unsqueeze(-1)
		chunk_size = max(1, self.predict_chunk_size)
		score_chunks = []
		for chunk in item_embed.split(chunk_size, dim=0):
			base = torch.matmul(interests, chunk.t())
			logits = base / self.temperature + prior_log
			weights = torch.softmax(logits, dim=1)
			score_chunks.append(torch.sum(weights * base, dim=1))
		return torch.cat(score_chunks, dim=1)

	def forward(self, *inputs: torch.Tensor, mode: str):
		if mode == "train_cf":
			return self.calc_cf_loss(*inputs)
		if mode == "predict":
			return self.calc_score(*inputs)
		raise ValueError(f"Unsupported mode: {mode}")
