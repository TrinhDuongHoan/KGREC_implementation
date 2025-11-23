from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBFM(nn.Module):
	def __init__(
		self,
		args,
		n_users: int,
		n_items: int,
		context_field_dims: Optional[Iterable[int]] = None,
	) -> None:
		super().__init__()
		context_field_dims = list(context_field_dims or [])

		self.n_users = n_users
		self.n_items = n_items
		self.context_field_dims = context_field_dims

		self.embed_dim = int(getattr(args, "embed_dim", 64))
		self.l2_reg = float(getattr(args, "l2_reg", 1e-6))
		self.use_sigmoid = bool(getattr(args, "use_sigmoid", False))

		self.global_bias = nn.Parameter(torch.zeros(1))
		self.user_linear = nn.Embedding(n_users, 1)
		self.item_linear = nn.Embedding(n_items, 1)
		self.context_linear = nn.ModuleList(
			nn.Embedding(field_dim, 1) for field_dim in context_field_dims
		)

		self.user_embed = nn.Embedding(n_users, self.embed_dim)
		self.item_embed = nn.Embedding(n_items, self.embed_dim)
		self.context_embed = nn.ModuleList(
			nn.Embedding(field_dim, self.embed_dim) for field_dim in context_field_dims
		)

		self.dropout_prob = float(getattr(args, "interaction_dropout", 0.0))
		self.interaction_dropout = nn.Dropout(self.dropout_prob)

		self._init_parameters()

	def _init_parameters(self) -> None:
		nn.init.zeros_(self.user_linear.weight)
		nn.init.zeros_(self.item_linear.weight)
		for emb in self.context_linear:
			nn.init.zeros_(emb.weight)

		nn.init.xavier_uniform_(self.user_embed.weight)
		nn.init.xavier_uniform_(self.item_embed.weight)
		for emb in self.context_embed:
			nn.init.xavier_uniform_(emb.weight)

	def _prepare_context(self, context_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
		if len(self.context_embed) == 0:
			return None
		if context_ids is None:
			raise ValueError("context_ids must be provided when context_field_dims is non-empty")
		if context_ids.dim() == 1:
			context_ids = context_ids.unsqueeze(-1)
		return context_ids.long()

	def _context_linear_term(self, context_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
		if len(self.context_linear) == 0:
			return None
		if context_ids is None:
			raise ValueError("context_ids must be provided when context_field_dims is non-empty")
		bias = torch.zeros(context_ids.size(0), device=context_ids.device)
		for idx, emb in enumerate(self.context_linear):
			bias = bias + emb(context_ids[:, idx]).squeeze(-1)
		return bias

	def _stack_user_fields(self, user_ids: torch.Tensor, context_ids: Optional[torch.Tensor], apply_dropout: bool) -> torch.Tensor:
		fields = [self.user_embed(user_ids)]
		if len(self.context_embed) > 0:
			if context_ids is None:
				raise ValueError("context_ids must be provided when context_field_dims is non-empty")
			for idx, emb in enumerate(self.context_embed):
				fields.append(emb(context_ids[:, idx]))
		stacked = torch.stack(fields, dim=1)
		if apply_dropout and self.dropout_prob > 0.0:
			stacked = self.interaction_dropout(stacked)
		return stacked

	def _pairwise_score(self, user_ids: torch.Tensor, item_ids: torch.Tensor, context_ids: Optional[torch.Tensor]) -> torch.Tensor:
		context_ids = self._prepare_context(context_ids) if len(self.context_embed) > 0 else None

		user_bias = self.user_linear(user_ids).squeeze(-1)
		if len(self.context_linear) > 0:
			user_bias = user_bias + self._context_linear_term(context_ids)

		item_bias = self.item_linear(item_ids).squeeze(-1)
		score = self.global_bias + user_bias + item_bias

		user_fields = self._stack_user_fields(user_ids, context_ids, apply_dropout=self.training)
		user_sum = user_fields.sum(dim=1)
		user_square = (user_fields * user_fields).sum(dim=1)
		user_user_term = 0.5 * ((user_sum * user_sum - user_square).sum(dim=1))

		item_embed = self.item_embed(item_ids)
		score = score + user_user_term + torch.sum(user_sum * item_embed, dim=1)
		return score

	def _matrix_score(self, user_ids: torch.Tensor, item_ids: torch.Tensor, context_ids: Optional[torch.Tensor]) -> torch.Tensor:
		context_ids = self._prepare_context(context_ids) if len(self.context_embed) > 0 else None

		user_bias = self.user_linear(user_ids).squeeze(-1)
		item_bias = self.item_linear(item_ids).squeeze(-1)
		score = self.global_bias + user_bias.unsqueeze(1) + item_bias.unsqueeze(0)

		if len(self.context_linear) > 0:
			score = score + self._context_linear_term(context_ids).unsqueeze(1)

		user_fields = self._stack_user_fields(user_ids, context_ids, apply_dropout=False)
		user_sum = user_fields.sum(dim=1)
		user_square = (user_fields * user_fields).sum(dim=1)
		user_user_term = 0.5 * ((user_sum * user_sum - user_square).sum(dim=1))
		score = score + user_user_term.unsqueeze(1)

		item_embed = self.item_embed(item_ids)
		score = score + torch.matmul(user_sum, item_embed.t())
		return score

	def _l2_norm(self, *tensors: torch.Tensor) -> torch.Tensor:
		reg = torch.zeros([], device=tensors[0].device)
		for t in tensors:
			reg = reg + t.pow(2).sum()
		return reg

	def calc_bpr_loss(
		self,
		user_ids: torch.Tensor,
		pos_item_ids: torch.Tensor,
		neg_item_ids: torch.Tensor,
		pos_context_ids: Optional[torch.Tensor] = None,
		neg_context_ids: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		pos_scores = self._pairwise_score(user_ids, pos_item_ids, pos_context_ids)
		neg_scores = self._pairwise_score(user_ids, neg_item_ids, neg_context_ids)
		loss = -F.logsigmoid(pos_scores - neg_scores).mean()
		if self.l2_reg > 0:
			user_vec = self.user_embed(user_ids)
			pos_vec = self.item_embed(pos_item_ids)
			neg_vec = self.item_embed(neg_item_ids)
			reg_tensors = [user_vec, pos_vec, neg_vec]
			if pos_context_ids is not None and len(self.context_embed) > 0:
				for idx, emb in enumerate(self.context_embed):
					reg_tensors.append(emb(pos_context_ids[:, idx]))
			if neg_context_ids is not None and len(self.context_embed) > 0:
				for idx, emb in enumerate(self.context_embed):
					reg_tensors.append(emb(neg_context_ids[:, idx]))
			reg = self._l2_norm(*reg_tensors)
			loss = loss + self.l2_reg * reg / user_ids.size(0)
		return loss

	def calc_pointwise_loss(
		self,
		user_ids: torch.Tensor,
		item_ids: torch.Tensor,
		context_ids: Optional[torch.Tensor],
		target: torch.Tensor,
		loss_type: str = "mse",
	) -> torch.Tensor:
		logits = self._pairwise_score(user_ids, item_ids, context_ids)
		loss_type = (loss_type or "mse").lower()
		if loss_type == "mse":
			preds = logits if not self.use_sigmoid else torch.sigmoid(logits)
			loss_val = F.mse_loss(preds, target)
		elif loss_type == "bce":
			preds = torch.sigmoid(logits)
			loss_val = F.binary_cross_entropy(preds, target)
		else:
			raise ValueError(f"Unsupported loss_type: {loss_type}")
		if self.l2_reg > 0:
			reg_tensors = [self.user_embed(user_ids), self.item_embed(item_ids)]
			if context_ids is not None and len(self.context_embed) > 0:
				for idx, emb in enumerate(self.context_embed):
					reg_tensors.append(emb(context_ids[:, idx]))
			reg = self._l2_norm(*reg_tensors)
			loss_val = loss_val + self.l2_reg * reg / user_ids.size(0)
		return loss_val

	def forward(self, *inputs, mode: str):
		if mode == "predict":
			logits = self._matrix_score(*inputs)
			return torch.sigmoid(logits) if self.use_sigmoid else logits
		if mode == "train_bpr":
			return self.calc_bpr_loss(*inputs)
		if mode == "train_pointwise":
			if not (4 <= len(inputs) <= 5):
				raise ValueError("train_pointwise expects 4 or 5 inputs: user, item, context, target, [loss_type]")
			loss_type = inputs[4] if len(inputs) == 5 else "mse"
			return self.calc_pointwise_loss(inputs[0], inputs[1], inputs[2], inputs[3], loss_type)
		raise ValueError(f"Unsupported mode: {mode}")
