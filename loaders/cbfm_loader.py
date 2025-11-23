from __future__ import annotations
import os
from typing import Iterable, Optional
import numpy as np
import torch
from loaders.base_loader import DataLoaderBase

class DataLoaderCBFM(DataLoaderBase):
	def __init__(self, args, logging):
		super().__init__(args, logging)
		self.cf_batch_size = int(getattr(args, "cf_batch_size", 1024))
		self.test_batch_size = int(getattr(args, "test_batch_size", 10000))
		self.eval_user_limit = int(getattr(args, "eval_user_limit", 1000))

		context_dims = getattr(args, "context_field_dims", [])
		if isinstance(context_dims, str):
			try:
				context_dims = eval(context_dims)
			except Exception as exc:
				raise ValueError(f"Invalid context_field_dims string: {context_dims}") from exc
		self.context_field_dims = list(context_dims) if isinstance(context_dims, Iterable) else []

		context_path = getattr(args, "context_path", None)
		self.user_context: Optional[torch.LongTensor] = None
		if self.context_field_dims:
			if context_path is not None:
				context_path = self._resolve_path(context_path)
				if os.path.exists(context_path):
					context_array = np.load(context_path)
					if context_array.shape != (self.n_users, len(self.context_field_dims)):
						raise ValueError(
							f"context array shape {context_array.shape} does not match "
							f"(n_users={self.n_users}, n_fields={len(self.context_field_dims)})"
						)
					context_array = np.asarray(context_array, dtype=np.int64)
				else:
					logging.warning(
						f"CBFM context_path '{context_path}' not found. Falling back to zero context."
					)
					context_array = np.zeros((self.n_users, len(self.context_field_dims)), dtype=np.int64)
			else:
				logging.info("CBFM: no context_path specified, defaulting to zero context features.")
				context_array = np.zeros((self.n_users, len(self.context_field_dims)), dtype=np.int64)

			self.user_context = torch.from_numpy(context_array)

		logging.info("CBFM DataLoader | cf_batch_size=%d | test_batch_size=%d | context_fields=%d",
					 self.cf_batch_size, self.test_batch_size, len(self.context_field_dims))

	def _resolve_path(self, path: str) -> str:
		if os.path.isabs(path):
			return path
		return os.path.join(self.data_dir, path)

	def get_user_context(self, user_ids) -> Optional[torch.LongTensor]:
		if self.user_context is None:
			return None
		if isinstance(user_ids, torch.Tensor):
			index = user_ids.detach().cpu().long()
		else:
			index = torch.as_tensor(user_ids, dtype=torch.long)
		return self.user_context.index_select(0, index)
