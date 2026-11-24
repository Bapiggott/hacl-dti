#!/usr/bin/env python3
"""V13 graph + longitudinal-metadata classifier.

This version preserves the strong graph-temporal architecture while adding a
separate history-to-query metadata path.  The history representation is loaded
through the project's existing cached-embedding tensors, but is no longer
forced into a generic early-fusion projector.

Main additions over V2:
  * independent history temporal branch;
  * late, early, both, or graph-to-history cross-attention fusion;
  * reliability-aware residual gates initialized near graph-only behavior;
  * optional modality dropout to prevent the graph branch from depending on
    metadata availability;
  * larger residual classification head, while retaining ``classifier`` for cRT.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from metadata_imputer import (
    BaselineAwareGRUImputer,
    BaselineAwareMaskedTransformerImputer,
    load_imputer_checkpoint,
)
from DTISequenceClassifier import (
    BaselineMetadataEncoder,
    MetadataTemporalBranch,
    VisitMetadataEncoder,
    VisitMetadataTransformerEncoder,
    build_metadata_fusion,
)
from DTISequenceClassifierV2 import TemporalAttentionBlock


class HistoryEmbeddingProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float,
        stats_dim: int = 8,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.stats_dim = min(int(stats_dim), max(self.input_dim, 0))
        hidden = max(output_dim * 2, 128)
        self.projector = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        reliability_input = self.stats_dim if self.stats_dim > 0 else self.input_dim
        self.reliability = nn.Sequential(
            nn.Linear(reliability_input, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        values: torch.Tensor,
        present: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.projector(values.float())
        if self.stats_dim > 0:
            reliability_input = values[..., -self.stats_dim :].float()
        else:
            reliability_input = values.float()
        reliability = self.reliability(reliability_input)
        if present is not None:
            present_float = present.float()
            projected = projected * present_float
            reliability = reliability * present_float
        return projected, reliability


class AvailabilityGatedFusion(nn.Module):
    """Residual multimodal fusion that begins close to graph-only behavior."""

    def __init__(
        self,
        graph_dim: int,
        metadata_dim: int,
        dropout: float,
        gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.metadata_projection = nn.Sequential(
            nn.Linear(metadata_dim, graph_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Linear(graph_dim * 2 + 1, graph_dim)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_init))
        self.norm = nn.LayerNorm(graph_dim)

    def forward(
        self,
        graph_hidden: torch.Tensor,
        metadata_hidden: torch.Tensor,
        reliability: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        metadata = self.metadata_projection(metadata_hidden)
        if reliability is None:
            reliability = torch.ones(
                graph_hidden.size(0),
                1,
                device=graph_hidden.device,
                dtype=graph_hidden.dtype,
            )
        gate_input = torch.cat([graph_hidden, metadata, reliability], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input)) * reliability
        return self.norm(graph_hidden + gate * metadata)


class CrossHistoryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        gate_init: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        self.query_norm = nn.LayerNorm(d_model)
        self.history_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(d_model * 2 + 1, d_model)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_init))
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        graph_tokens: torch.Tensor,
        history_tokens: torch.Tensor,
        graph_pad_mask: torch.Tensor,
        history_present: Optional[torch.Tensor],
        history_reliability: Optional[torch.Tensor],
    ) -> torch.Tensor:
        history_mask = graph_pad_mask.clone()
        if history_present is not None:
            history_mask = history_mask | (history_present.squeeze(-1) <= 0)
        all_missing = history_mask.all(dim=1)
        safe_history_mask = history_mask.clone()
        safe_history_mask[all_missing, 0] = False

        attended, _ = self.attention(
            self.query_norm(graph_tokens),
            self.history_norm(history_tokens),
            self.history_norm(history_tokens),
            key_padding_mask=safe_history_mask,
            need_weights=False,
        )
        attended[all_missing] = 0.0

        if history_reliability is None:
            reliability = (~all_missing).to(graph_tokens.dtype).unsqueeze(-1)
            reliability = reliability.unsqueeze(1).expand(-1, graph_tokens.size(1), -1)
        else:
            reliability = history_reliability

        gate = torch.sigmoid(
            self.gate(torch.cat([graph_tokens, attended, reliability], dim=-1))
        ) * reliability
        output = self.output_norm(graph_tokens + gate * self.dropout(attended))
        return output.masked_fill(graph_pad_mask.unsqueeze(-1), 0.0)


class ResidualClassifierHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.residual = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
            nn.Dropout(dropout),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        x = x + self.residual(normalized)
        return self.output(x)


class DTISequenceClassifierV13(nn.Module):
    def __init__(
        self,
        graph_emb_dim: int = 512,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.5,
        bidirectional: bool = True,
        num_classes: int = 2,
        rnn_type: str = "GRU",
        use_prev_dx: bool = False,
        dx_emb_dim: int = 8,
        use_metadata: bool = False,
        meta_num_dim: int = 0,
        meta_cat_cardinalities: Optional[List[int]] = None,
        meta_emb_dim: int = 64,
        meta_encoder_type: str = "ft_transformer",
        meta_token_dim: int = 32,
        meta_transformer_layers: int = 2,
        meta_transformer_heads: int = 4,
        use_metadata_branch: bool = False,
        metadata_branch_dim: int = 128,
        metadata_branch_layers: int = 1,
        metadata_branch_heads: int = 2,
        metadata_branch_dropout: float = 0.4,
        metadata_fusion_operator: str = "gated",
        metadata_include_baseline: bool = False,
        metadata_imputer_mode: str = "none",
        metadata_imputer_architecture: str = "transformer",
        metadata_imputer_checkpoint: Optional[str] = None,
        metadata_imputer_d_model: int = 128,
        metadata_imputer_layers: int = 2,
        metadata_imputer_heads: int = 4,
        metadata_imputer_dropout: float = 0.2,
        metadata_imputer_mask_probability: float = 0.2,
        metadata_imputer_visit_mask_probability: float = 0.05,
        use_ollama_metadata: bool = False,
        ollama_input_dim: int = 0,
        ollama_emb_dim: int = 128,
        use_time_gap_input: bool = False,
        n_attn_heads: int = 2,
        history_fusion: str = "late",
        history_branch_dim: int = 128,
        history_branch_layers: int = 2,
        history_branch_heads: int = 4,
        history_dropout: float = 0.3,
        history_gate_init: float = -2.0,
        history_stats_dim: int = 8,
        classifier_head: str = "large",
        metadata_modality_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        del num_layers, bidirectional, rnn_type
        self.graph_emb_dim = int(graph_emb_dim)
        self.use_prev_dx = bool(use_prev_dx)
        self.use_metadata = bool(use_metadata)
        self.use_metadata_branch = bool(use_metadata_branch)
        self.use_history = bool(use_ollama_metadata and ollama_input_dim > 0)
        self.use_time_gap_input = bool(use_time_gap_input)
        self.metadata_include_baseline = bool(metadata_include_baseline)
        self.metadata_imputer_mode = str(metadata_imputer_mode).lower()
        self.metadata_imputer_mask_probability = float(
            metadata_imputer_mask_probability
        )
        self.metadata_imputer_visit_mask_probability = float(
            metadata_imputer_visit_mask_probability
        )
        self.history_fusion = str(history_fusion).lower()
        self.metadata_modality_dropout = float(metadata_modality_dropout)
        self._imputation_loss = torch.tensor(0.0)

        valid_history_fusions = {"none", "early", "late", "both", "cross"}
        if self.history_fusion not in valid_history_fusions:
            raise ValueError(
                f"history_fusion must be one of {sorted(valid_history_fusions)}"
            )
        if self.history_fusion != "none" and not self.use_history:
            raise ValueError(
                "History fusion is enabled but no history cache embedding is available"
            )

        input_dim = self.graph_emb_dim
        if self.use_prev_dx:
            self.dx_embedding = nn.Embedding(3, dx_emb_dim)
            input_dim += dx_emb_dim
        else:
            self.dx_embedding = None

        needs_structured_metadata = self.use_metadata or self.use_metadata_branch
        if needs_structured_metadata:
            encoder_type = str(meta_encoder_type).lower()
            if encoder_type in {"ft_transformer", "transformer", "ft_plus_ollama"}:
                self.meta_encoder = VisitMetadataTransformerEncoder(
                    meta_num_dim,
                    meta_cat_cardinalities or [],
                    out_dim=meta_emb_dim,
                    token_dim=meta_token_dim,
                    n_layers=meta_transformer_layers,
                    n_heads=meta_transformer_heads,
                    dropout=dropout,
                )
            elif encoder_type == "mlp":
                self.meta_encoder = VisitMetadataEncoder(
                    meta_num_dim,
                    meta_cat_cardinalities or [],
                    out_dim=meta_emb_dim,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unknown metadata encoder: {meta_encoder_type!r}")
        else:
            self.meta_encoder = None

        if needs_structured_metadata and self.metadata_include_baseline:
            self.baseline_encoder = BaselineMetadataEncoder(
                meta_num_dim, meta_emb_dim, dropout
            )
            self.metadata_combiner = nn.Sequential(
                nn.Linear(meta_emb_dim * 2, meta_emb_dim),
                nn.LayerNorm(meta_emb_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.baseline_encoder = None
            self.metadata_combiner = None

        self.metadata_imputer = None
        if needs_structured_metadata and self.metadata_imputer_mode != "none":
            if metadata_imputer_checkpoint:
                imputer, _checkpoint = load_imputer_checkpoint(
                    metadata_imputer_checkpoint, map_location="cpu"
                )
                if imputer.feature_dim != int(meta_num_dim):
                    raise ValueError(
                        f"Imputer feature_dim={imputer.feature_dim}, "
                        f"dataset metadata dimension={meta_num_dim}"
                    )
                self.metadata_imputer = imputer
            else:
                imputer_class = (
                    BaselineAwareGRUImputer
                    if str(metadata_imputer_architecture).lower() == "gru"
                    else BaselineAwareMaskedTransformerImputer
                )
                self.metadata_imputer = imputer_class(
                    feature_dim=meta_num_dim,
                    d_model=metadata_imputer_d_model,
                    n_layers=metadata_imputer_layers,
                    n_heads=metadata_imputer_heads,
                    dropout=metadata_imputer_dropout,
                    use_baseline=True,
                )
            if self.metadata_imputer_mode == "frozen":
                for parameter in self.metadata_imputer.parameters():
                    parameter.requires_grad_(False)
                self.metadata_imputer.eval()

        if self.use_metadata:
            input_dim += meta_emb_dim

        if self.use_metadata_branch:
            self.metadata_branch = MetadataTemporalBranch(
                input_dim=meta_emb_dim,
                branch_dim=metadata_branch_dim,
                n_layers=metadata_branch_layers,
                n_heads=metadata_branch_heads,
                dropout=metadata_branch_dropout,
            )
        else:
            self.metadata_branch = None

        if self.use_history:
            self.history_projector = HistoryEmbeddingProjector(
                input_dim=ollama_input_dim,
                output_dim=ollama_emb_dim,
                dropout=history_dropout,
                stats_dim=history_stats_dim,
            )
            if self.history_fusion in {"early", "both"}:
                input_dim += ollama_emb_dim
            if self.history_fusion in {"late", "both", "cross"}:
                self.history_branch = MetadataTemporalBranch(
                    input_dim=ollama_emb_dim,
                    branch_dim=history_branch_dim,
                    n_layers=history_branch_layers,
                    n_heads=history_branch_heads,
                    dropout=history_dropout,
                )
            else:
                self.history_branch = None
        else:
            self.history_projector = None
            self.history_branch = None

        if self.use_time_gap_input:
            self.time_projection = nn.Sequential(
                nn.Linear(1, 8),
                nn.GELU(),
                nn.Linear(8, 8),
                nn.GELU(),
            )
            input_dim += 8
        else:
            self.time_projection = None

        d_model = hidden_dim * 2
        self.d_model = d_model
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.attn_block = TemporalAttentionBlock(
            d_model, n_heads=n_attn_heads, dropout=dropout
        )
        self.pool_gate = nn.Linear(d_model, 1, bias=False)

        if self.history_fusion == "cross":
            self.history_cross_projection = nn.Linear(ollama_emb_dim, d_model)
            self.history_cross_attention = CrossHistoryAttention(
                d_model,
                n_heads=n_attn_heads,
                dropout=history_dropout,
                gate_init=history_gate_init,
            )
        else:
            self.history_cross_projection = None
            self.history_cross_attention = None

        if self.use_metadata_branch:
            self.metadata_fusion = build_metadata_fusion(
                metadata_fusion_operator,
                graph_dim=d_model,
                metadata_dim=metadata_branch_dim,
                dropout=metadata_branch_dropout,
            )
        else:
            self.metadata_fusion = None

        if self.history_branch is not None:
            self.history_late_fusion = AvailabilityGatedFusion(
                graph_dim=d_model,
                metadata_dim=history_branch_dim,
                dropout=history_dropout,
                gate_init=history_gate_init,
            )
        else:
            self.history_late_fusion = None

        head_type = str(classifier_head).lower()
        if head_type == "small":
            self.classifier = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes),
            )
        elif head_type == "large":
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )
        elif head_type == "residual":
            self.classifier = ResidualClassifierHead(
                d_model, num_classes=num_classes, dropout=dropout
            )
        else:
            raise ValueError(f"Unknown classifier head: {classifier_head!r}")

        self._last_hidden = None
        self._return_hidden = False
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Keep intentionally initialized gate biases.
                if module is getattr(self, "pool_gate", None):
                    continue
                if module.bias is not None and torch.all(module.bias == module.bias[0]):
                    pass
                nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(self.pool_gate.weight)

    def _apply_modality_dropout(
        self,
        embeddings: Optional[torch.Tensor],
        present: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if (
            embeddings is None
            or not self.training
            or self.metadata_modality_dropout <= 0.0
        ):
            return embeddings, present
        batch_size = embeddings.size(0)
        keep = (
            torch.rand(batch_size, 1, 1, device=embeddings.device)
            >= self.metadata_modality_dropout
        ).to(embeddings.dtype)
        embeddings = embeddings * keep
        if present is not None:
            present = present * keep
        return embeddings, present

    def forward(
        self,
        graph_seq: torch.Tensor,
        lengths: torch.Tensor,
        prev_dx_seq: Optional[torch.Tensor] = None,
        meta_num_seq: Optional[torch.Tensor] = None,
        meta_mask_seq: Optional[torch.Tensor] = None,
        meta_cat_seq: Optional[torch.Tensor] = None,
        meta_present_seq: Optional[torch.Tensor] = None,
        meta_baseline_num_seq: Optional[torch.Tensor] = None,
        meta_baseline_mask_seq: Optional[torch.Tensor] = None,
        meta_months_since_baseline_seq: Optional[torch.Tensor] = None,
        meta_months_since_previous_seq: Optional[torch.Tensor] = None,
        ollama_seq: Optional[torch.Tensor] = None,
        ollama_present_seq: Optional[torch.Tensor] = None,
        time_gaps_raw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, max_visits, _ = graph_seq.shape
        pieces = [graph_seq]

        if self.use_prev_dx:
            if prev_dx_seq is None:
                raise ValueError("use_prev_dx=True but prev_dx_seq was not provided")
            pieces.append(self.dx_embedding(prev_dx_seq))

        metadata_visits = None
        metadata_temporal_present = meta_present_seq
        self._imputation_loss = graph_seq.sum() * 0.0
        if self.use_metadata or self.use_metadata_branch:
            if meta_num_seq is None or meta_mask_seq is None:
                raise ValueError("Structured metadata tensors were not provided")
            metadata_values = meta_num_seq
            if self.metadata_imputer is not None:
                compute_reconstruction = (
                    self.training
                    and self.metadata_imputer_mode in {"joint", "scratch_joint"}
                )
                imputed = self.metadata_imputer(
                    meta_num_seq,
                    meta_mask_seq,
                    lengths,
                    baseline_values=meta_baseline_num_seq,
                    baseline_mask=meta_baseline_mask_seq,
                    months_since_baseline=meta_months_since_baseline_seq,
                    months_since_previous=meta_months_since_previous_seq,
                    mask_probability=self.metadata_imputer_mask_probability,
                    visit_mask_probability=self.metadata_imputer_visit_mask_probability,
                    compute_reconstruction=compute_reconstruction,
                )
                metadata_values = imputed.filled
                self._imputation_loss = imputed.reconstruction_loss

            encoder_present = meta_present_seq
            if self.metadata_imputer is not None and meta_baseline_mask_seq is not None:
                baseline_any = meta_baseline_mask_seq.any(dim=-1, keepdim=True).to(
                    metadata_values.dtype
                )
                encoder_present = (
                    baseline_any
                    if encoder_present is None
                    else torch.maximum(encoder_present.float(), baseline_any)
                )

            metadata_visits = self.meta_encoder(
                metadata_values,
                meta_mask_seq,
                meta_cat_seq,
                encoder_present,
            )
            if self.baseline_encoder is not None:
                baseline_values = (
                    meta_baseline_num_seq
                    if meta_baseline_num_seq is not None
                    else torch.zeros_like(meta_num_seq)
                )
                baseline_masks = (
                    meta_baseline_mask_seq
                    if meta_baseline_mask_seq is not None
                    else torch.zeros_like(meta_mask_seq)
                )
                baseline_visits = self.baseline_encoder(
                    baseline_values,
                    baseline_masks,
                    meta_months_since_baseline_seq,
                )
                metadata_visits = self.metadata_combiner(
                    torch.cat([metadata_visits, baseline_visits], dim=-1)
                )
                baseline_present = baseline_masks.any(dim=-1, keepdim=True).to(
                    metadata_visits.dtype
                )
                metadata_temporal_present = (
                    baseline_present
                    if meta_present_seq is None
                    else torch.maximum(meta_present_seq.float(), baseline_present)
                )
                metadata_visits = metadata_visits * metadata_temporal_present

            metadata_visits, metadata_temporal_present = self._apply_modality_dropout(
                metadata_visits, metadata_temporal_present
            )
            if self.use_metadata:
                pieces.append(metadata_visits)

        history_visits = None
        history_reliability = None
        history_present = ollama_present_seq
        if self.use_history:
            if ollama_seq is None:
                raise ValueError("History cache is enabled but ollama_seq is absent")
            history_visits, history_reliability = self.history_projector(
                ollama_seq, history_present
            )
            history_visits, history_present = self._apply_modality_dropout(
                history_visits, history_present
            )
            if history_reliability is not None and history_present is not None:
                history_reliability = history_reliability * history_present.float()
            if self.history_fusion in {"early", "both"}:
                pieces.append(history_visits)

        if self.use_time_gap_input:
            time_value = (
                torch.zeros(batch_size, 1, device=graph_seq.device)
                if time_gaps_raw is None
                else time_gaps_raw.view(batch_size, 1)
            )
            time_embedding = self.time_projection(time_value).unsqueeze(1).expand(
                batch_size, max_visits, -1
            )
            pieces.append(time_embedding)

        x = self.input_proj(torch.cat(pieces, dim=-1))
        positions = torch.arange(max_visits, device=graph_seq.device).unsqueeze(0)
        pad_mask = positions >= lengths.to(graph_seq.device).unsqueeze(1)
        x = self.attn_block(x, key_padding_mask=pad_mask)

        if self.history_fusion == "cross":
            assert history_visits is not None
            history_for_cross = self.history_cross_projection(history_visits)
            x = self.history_cross_attention(
                x,
                history_for_cross,
                graph_pad_mask=pad_mask,
                history_present=history_present,
                history_reliability=history_reliability,
            )

        gate_scores = self.pool_gate(x).squeeze(-1).masked_fill(
            pad_mask, float("-inf")
        )
        visit_weights = F.softmax(gate_scores, dim=-1).unsqueeze(-1)
        final_hidden = (x * visit_weights).sum(dim=1)

        if self.use_metadata_branch:
            assert metadata_visits is not None
            metadata_hidden = self.metadata_branch(
                metadata_visits,
                lengths,
                metadata_temporal_present,
            )
            final_hidden = self.metadata_fusion(final_hidden, metadata_hidden)

        if self.history_branch is not None:
            assert history_visits is not None
            history_hidden = self.history_branch(
                history_visits,
                lengths,
                history_present,
            )
            if history_reliability is None:
                pooled_reliability = None
            else:
                valid = (~pad_mask).unsqueeze(-1).to(history_reliability.dtype)
                if history_present is not None:
                    valid = valid * history_present.float()
                denominator = valid.sum(dim=1).clamp_min(1.0)
                pooled_reliability = (
                    (history_reliability * valid).sum(dim=1) / denominator
                )
            final_hidden = self.history_late_fusion(
                final_hidden,
                history_hidden,
                pooled_reliability,
            )

        self._last_hidden = final_hidden
        if self._return_hidden:
            return final_hidden
        return self.classifier(final_hidden)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.metadata_imputer is not None and self.metadata_imputer_mode == "frozen":
            self.metadata_imputer.eval()
        return self

    def set_return_hidden(self, flag: bool = True) -> None:
        self._return_hidden = bool(flag)

    @property
    def hidden_dim(self) -> int:
        return self.d_model
