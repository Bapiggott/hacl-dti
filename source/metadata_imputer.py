#!/usr/bin/env python3
"""Baseline-aware masked Transformer imputer for longitudinal ADNI metadata.

This is intentionally compact enough for an RTX 2080 Ti and short ADNI visit
sequences. It is a SAITS-inspired denoising Transformer rather than a verbatim
reimplementation of SAITS.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ImputerOutput:
    predictions: torch.Tensor
    filled: torch.Tensor
    reconstruction_loss: torch.Tensor
    artificial_mask: torch.Tensor


class BaselineAwareMaskedTransformerImputer(nn.Module):
    """Impute numerical metadata using visit history, masks, and baseline values.

    Inputs are shaped [B, T, F]. Values are expected to be standardized. Missing
    values should be zero with masks equal to zero.
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.20,
        use_baseline: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.feature_dim = int(feature_dim)
        self.d_model = int(d_model)
        self.use_baseline = bool(use_baseline)

        # Current values + masks, optional baseline values + masks, and two time channels.
        in_dim = self.feature_dim * 2 + 2
        if self.use_baseline:
            in_dim += self.feature_dim * 2

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.position_scale = nn.Parameter(torch.tensor(1.0))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.feature_dim),
        )
        self.feature_bias = nn.Parameter(torch.zeros(self.feature_dim))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _normalize_time(x: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
        if x is None:
            return torch.zeros((*like.shape[:2], 1), dtype=like.dtype, device=like.device)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        # log1p keeps long gaps from dominating while preserving ordering.
        return torch.log1p(torch.clamp(x.to(like.dtype), min=0.0)) / 5.0

    def predict(
        self,
        values: torch.Tensor,
        observed_mask: torch.Tensor,
        lengths: torch.Tensor,
        baseline_values: Optional[torch.Tensor] = None,
        baseline_mask: Optional[torch.Tensor] = None,
        months_since_baseline: Optional[torch.Tensor] = None,
        months_since_previous: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if values.ndim != 3 or observed_mask.shape != values.shape:
            raise ValueError("values and observed_mask must both be [B,T,F]")
        if values.size(-1) != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, received {values.size(-1)}"
            )

        pieces = [values, observed_mask]
        if self.use_baseline:
            if baseline_values is None:
                baseline_values = torch.zeros_like(values)
            if baseline_mask is None:
                baseline_mask = torch.zeros_like(observed_mask)
            pieces.extend([baseline_values, baseline_mask])

        pieces.append(self._normalize_time(months_since_baseline, values))
        pieces.append(self._normalize_time(months_since_previous, values))
        x = self.input_proj(torch.cat(pieces, dim=-1))

        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device, dtype=x.dtype).view(1, T, 1)
        x = x + self.position_scale * positions / max(T, 1)
        pad_mask = torch.arange(T, device=x.device).view(1, T) >= lengths.to(x.device).view(B, 1)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        pred = self.output_head(x) + self.feature_bias
        return pred.masked_fill(pad_mask.unsqueeze(-1), 0.0)

    def forward(
        self,
        values: torch.Tensor,
        observed_mask: torch.Tensor,
        lengths: torch.Tensor,
        baseline_values: Optional[torch.Tensor] = None,
        baseline_mask: Optional[torch.Tensor] = None,
        months_since_baseline: Optional[torch.Tensor] = None,
        months_since_previous: Optional[torch.Tensor] = None,
        mask_probability: float = 0.0,
        visit_mask_probability: float = 0.0,
        compute_reconstruction: bool = False,
    ) -> ImputerOutput:
        original_values = values
        original_mask = observed_mask
        artificial_mask = torch.zeros_like(original_mask, dtype=torch.bool)

        if compute_reconstruction and mask_probability > 0:
            valid = original_mask > 0.5
            random_draw = torch.rand_like(original_mask)
            artificial_mask = valid & (random_draw < float(mask_probability))
            if visit_mask_probability > 0:
                visit_draw = torch.rand((*original_mask.shape[:2], 1), device=original_mask.device)
                artificial_mask = artificial_mask | (valid & (visit_draw < float(visit_mask_probability)))
            # Ensure at least one target when the batch contains observed data.
            if valid.any() and not artificial_mask.any():
                first = valid.nonzero(as_tuple=False)[0]
                artificial_mask[tuple(first.tolist())] = True
            model_mask = original_mask.clone()
            model_mask[artificial_mask] = 0.0
            model_values = original_values.clone()
            model_values[artificial_mask] = 0.0
        else:
            model_values = original_values
            model_mask = original_mask

        predictions = self.predict(
            model_values,
            model_mask,
            lengths,
            baseline_values=baseline_values,
            baseline_mask=baseline_mask,
            months_since_baseline=months_since_baseline,
            months_since_previous=months_since_previous,
        )

        # Only truly missing fields are imputed for classifier input. Artificially
        # masked observed values are restored so classifier training is stable.
        filled = torch.where(original_mask > 0.5, original_values, predictions)

        if artificial_mask.any():
            diff = predictions[artificial_mask] - original_values[artificial_mask]
            reconstruction_loss = 0.5 * F.l1_loss(
                predictions[artificial_mask], original_values[artificial_mask]
            ) + 0.5 * torch.mean(diff.square())
        else:
            reconstruction_loss = predictions.sum() * 0.0

        return ImputerOutput(
            predictions=predictions,
            filled=filled,
            reconstruction_loss=reconstruction_loss,
            artificial_mask=artificial_mask,
        )


class BaselineAwareGRUImputer(BaselineAwareMaskedTransformerImputer):
    """Forward GRU denoising baseline with the same interface as the Transformer."""
    def __init__(
        self,
        feature_dim: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,  # accepted for checkpoint/config compatibility
        dropout: float = 0.20,
        use_baseline: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        del n_heads
        self.feature_dim = int(feature_dim)
        self.d_model = int(d_model)
        self.use_baseline = bool(use_baseline)
        in_dim = self.feature_dim * 2 + 2
        if self.use_baseline:
            in_dim += self.feature_dim * 2
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.gru = nn.GRU(
            input_size=d_model, hidden_size=d_model, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, self.feature_dim),
        )
        self.feature_bias = nn.Parameter(torch.zeros(self.feature_dim))
        self._reset_parameters()

    def predict(
        self, values, observed_mask, lengths, baseline_values=None, baseline_mask=None,
        months_since_baseline=None, months_since_previous=None,
    ):
        pieces = [values, observed_mask]
        if self.use_baseline:
            if baseline_values is None: baseline_values = torch.zeros_like(values)
            if baseline_mask is None: baseline_mask = torch.zeros_like(observed_mask)
            pieces.extend([baseline_values, baseline_mask])
        pieces.append(self._normalize_time(months_since_baseline, values))
        pieces.append(self._normalize_time(months_since_previous, values))
        x = self.input_proj(torch.cat(pieces, dim=-1))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=values.size(1)
        )
        pred = self.output_head(output) + self.feature_bias
        pad = torch.arange(values.size(1), device=values.device).view(1, -1) >= lengths.to(values.device).view(-1, 1)
        return pred.masked_fill(pad.unsqueeze(-1), 0.0)


def load_imputer_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> Tuple[BaselineAwareMaskedTransformerImputer, Dict]:
    checkpoint = torch.load(str(Path(checkpoint_path).expanduser()), map_location=map_location, weights_only=False)
    config = dict(checkpoint.get("model_config", {}))
    if "feature_dim" not in config:
        config["feature_dim"] = len(checkpoint.get("fields", []))
    architecture = str(checkpoint.get("architecture", "transformer")).lower()
    model_cls = BaselineAwareGRUImputer if architecture == "gru" else BaselineAwareMaskedTransformerImputer
    model = model_cls(**config)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint
