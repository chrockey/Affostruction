"""
CLIP text encoder used as the conditioning model for affordance heatmap flow.

Mirrors the training-time ``TextConditionedMixin`` (CLIPTextModel +
AutoTokenizer, frozen, eval-mode) but wraps it in a plain ``nn.Module`` so the
pipeline can manage device placement uniformly with the other models.
"""

from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel


class CLIPTextConditioner(nn.Module):
    """Frozen CLIP text encoder.

    ``encode([texts])`` returns ``(B, 77, hidden_dim)`` token embeddings
    matching the trainer's behavior. ``null_cond()`` returns the cached
    embedding of the empty string for classifier-free guidance.
    """

    MAX_LENGTH = 77

    def __init__(self, name: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        self.name = name
        self.model = CLIPTextModel.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._null_cond: Optional[torch.Tensor] = None

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def hidden_dim(self) -> int:
        return self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, text: List[str]) -> torch.Tensor:
        assert isinstance(text, list) and all(isinstance(t, str) for t in text), (
            "CLIPTextConditioner.encode expects a list[str]"
        )
        enc = self.tokenizer(
            text,
            max_length=self.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(self.device)
        return self.model(input_ids=ids).last_hidden_state

    @torch.no_grad()
    def null_cond(self) -> torch.Tensor:
        if self._null_cond is None or self._null_cond.device != self.device:
            self._null_cond = self.encode([""])
        return self._null_cond
