#!/usr/bin/env python3
"""
DTISequenceClassifierV2 — attention-based drop-in replacement for DTISequenceClassifier.

Key change: replaces BiGRU with:
  1. Linear projection:  input_dim → d_model (= hidden_dim * 2)
  2. 1 pre-norm transformer block (self-attention + FFN)
  3. Learned soft-pooling gate over the visit dimension

Output shape is identical to BiGRU: [B, hidden_dim * 2].
All other components (dx_embedding, meta_encoder, time_projection, classifier head)
are copied verbatim from DTISequenceClassifier so the calling interface is drop-in.

Why attention for short sequences (T=2–9, median T=3)?
  • The BiGRU processes visits left-to-right and compresses prior context into a
    fixed state vector. With T≤9 that state must summarise everything the model
    has seen so far — any early-visit information that isn't immediately useful
    gets over-written.
  • Attention gives every visit a direct path to every other visit, regardless of
    temporal distance. The learned pooling gate then decides which visits to
    up-weight for the final decision (e.g. the transition visit may matter more
    than stable baseline visits).
  • Fewer sequential inductive biases → harder to memorise ordering artefacts,
    better generalisation across subjects with different visit spacings.

Usage in v9 (via patch_v9_to_v10.py):
  --use_attention_temporal      swap DTISequenceClassifier → DTISequenceClassifierV2
  --attn_heads 2                number of attention heads (must divide hidden_dim*2)

Ignored args (accepted for API compatibility, not used):
  num_layers, rnn_type, bidirectional
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from longitudinal_innovation import VisitReliabilityFiLM

from metadata_imputer import BaselineAwareMaskedTransformerImputer, BaselineAwareGRUImputer, load_imputer_checkpoint

# Reuse helper encoders from the original file — no duplication.
from DTISequenceClassifier import (
    GatedMetadataFusion,
    BaselineMetadataEncoder,
    build_metadata_fusion,
    MetadataTemporalBranch,
    OllamaEmbeddingProjector,
    VisitMetadataEncoder,
    VisitMetadataTransformerEncoder,
)


# ---------------------------------------------------------------------------
# Temporal attention building block
# ---------------------------------------------------------------------------

class TemporalAttentionBlock(nn.Module):
    """Pre-norm transformer block for short visit sequences.

    Architecture:
        x → LN → MHA(x,x,x) → residual → LN → FFN → residual

    Pre-norm (norm_first) is empirically more stable at small dataset sizes
    than post-norm, and avoids the early-training instability seen with the
    original post-norm transformer.

    Args:
        d_model:  token dimension (= hidden_dim * 2 in this context)
        n_heads:  number of attention heads; must divide d_model evenly
        dropout:  applied inside MHA and FFN, and on both residual paths
    """

    def __init__(self, d_model: int, n_heads: int = 2, dropout: float = 0.3):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}). "
                f"With hidden_dim=128, d_model=256, use n_heads ∈ {{1,2,4,8,16,32}}."
            )
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # FFN: expand × 2 (lighter than the standard × 4 to avoid overfitting)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                [B, T, d_model]
            key_padding_mask: [B, T] bool, True at PAD positions (ignored in
                              attention and pooling).
        Returns:
            [B, T, d_model]
        """
        # Pre-norm self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop(h)
        # Pre-norm FFN
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# V2 classifier
# ---------------------------------------------------------------------------

class DTISequenceClassifierV2(nn.Module):
    """Attention-based temporal classifier with optional late metadata branch."""

    def __init__(
        self,
        graph_emb_dim: int = 512,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
        num_classes: int = 3,
        rnn_type: str = "GRU",
        use_prev_dx: bool = False,
        dx_emb_dim: int = 8,
        use_metadata: bool = False,
        meta_num_dim: int = 0,
        meta_cat_cardinalities: Optional[List[int]] = None,
        meta_emb_dim: int = 64,
        meta_encoder_type: str = "mlp",
        meta_token_dim: int = 32,
        meta_transformer_layers: int = 2,
        meta_transformer_heads: int = 4,
        use_metadata_branch: bool = False,
        metadata_branch_dim: int = 128,
        metadata_branch_layers: int = 1,
        metadata_branch_heads: int = 2,
        metadata_branch_dropout: float = 0.3,
        metadata_fusion_operator: str = "gated",
        metadata_include_baseline: bool = False,
        metadata_imputer_mode: str = "none",
        metadata_imputer_architecture: str = "transformer",
        metadata_imputer_checkpoint: Optional[str] = None,
        metadata_imputer_d_model: int = 128,
        metadata_imputer_layers: int = 2,
        metadata_imputer_heads: int = 4,
        metadata_imputer_dropout: float = 0.20,
        metadata_imputer_mask_probability: float = 0.20,
        metadata_imputer_visit_mask_probability: float = 0.05,
        use_ollama_metadata: bool = False,
        ollama_input_dim: int = 0,
        ollama_emb_dim: int = 64,
        use_time_gap_input: bool = False,
        n_attn_heads: int = 2,
        use_visit_film: bool = False,
        use_clinical_reliability: bool = False,
        reliability_tau_days: float = 90.0,
        clinical_reliability_mode: str = "hard",
        clinical_film_mode: str = "standard",
    ):
        super().__init__()
        del num_layers, bidirectional, rnn_type
        self.graph_emb_dim = int(graph_emb_dim)
        self.use_prev_dx = bool(use_prev_dx)
        self.use_metadata = bool(use_metadata)
        self.use_metadata_branch = bool(use_metadata_branch)
        self.use_ollama_metadata = bool(use_ollama_metadata)
        self.use_time_gap_input = bool(use_time_gap_input)
        self.use_visit_film = bool(use_visit_film)
        self.use_clinical_reliability = bool(use_clinical_reliability)
        if self.use_clinical_reliability and not self.use_visit_film:
            raise ValueError('Clinical reliability needs visit-level FiLM')
        if self.use_visit_film and not use_metadata_branch:
            raise ValueError('Visit FiLM requires the clinical temporal branch')
        self.meta_encoder_type = str(meta_encoder_type).lower()
        self.metadata_fusion_operator = str(metadata_fusion_operator).lower()
        self.metadata_include_baseline = bool(metadata_include_baseline)
        self.metadata_imputer_mode = str(metadata_imputer_mode).lower()
        self.metadata_imputer_architecture = str(metadata_imputer_architecture).lower()
        self.metadata_imputer_mask_probability = float(metadata_imputer_mask_probability)
        self.metadata_imputer_visit_mask_probability = float(metadata_imputer_visit_mask_probability)
        self._imputation_loss = torch.tensor(0.0)

        input_dim = self.graph_emb_dim
        if self.use_prev_dx:
            self.dx_embedding = nn.Embedding(3, dx_emb_dim)
            input_dim += dx_emb_dim
        else:
            self.dx_embedding = None

        needs_structured_metadata = self.use_metadata or self.use_metadata_branch
        if needs_structured_metadata:
            if self.meta_encoder_type in {"ft_transformer", "transformer", "ft_plus_ollama"}:
                self.meta_encoder = VisitMetadataTransformerEncoder(
                    meta_num_dim,
                    meta_cat_cardinalities or [],
                    out_dim=meta_emb_dim,
                    token_dim=meta_token_dim,
                    n_layers=meta_transformer_layers,
                    n_heads=meta_transformer_heads,
                    dropout=dropout,
                )
            elif self.meta_encoder_type == "mlp":
                self.meta_encoder = VisitMetadataEncoder(
                    meta_num_dim,
                    meta_cat_cardinalities or [],
                    out_dim=meta_emb_dim,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unknown meta_encoder_type: {meta_encoder_type!r}")
        else:
            self.meta_encoder = None

        if needs_structured_metadata and self.metadata_include_baseline:
            self.baseline_encoder = BaselineMetadataEncoder(meta_num_dim, meta_emb_dim, dropout)
            self.metadata_combiner = nn.Sequential(
                nn.Linear(meta_emb_dim * 2, meta_emb_dim), nn.LayerNorm(meta_emb_dim),
                nn.GELU(), nn.Dropout(dropout),
            )
        else:
            self.baseline_encoder = None
            self.metadata_combiner = None

        self.metadata_imputer = None
        if needs_structured_metadata and self.metadata_imputer_mode != "none":
            if metadata_imputer_checkpoint:
                imputer, checkpoint = load_imputer_checkpoint(metadata_imputer_checkpoint, map_location="cpu")
                if imputer.feature_dim != int(meta_num_dim):
                    raise ValueError(
                        f"Imputer feature_dim={imputer.feature_dim} but dataset meta_num_dim={meta_num_dim}"
                    )
                self.metadata_imputer = imputer
            else:
                imputer_cls = (
                    BaselineAwareGRUImputer
                    if self.metadata_imputer_architecture == "gru"
                    else BaselineAwareMaskedTransformerImputer
                )
                self.metadata_imputer = imputer_cls(
                    feature_dim=meta_num_dim, d_model=metadata_imputer_d_model,
                    n_layers=metadata_imputer_layers, n_heads=metadata_imputer_heads,
                    dropout=metadata_imputer_dropout, use_baseline=True,
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

        if self.use_ollama_metadata:
            self.ollama_encoder = OllamaEmbeddingProjector(
                input_dim=ollama_input_dim,
                out_dim=ollama_emb_dim,
                dropout=dropout,
            )
            input_dim += ollama_emb_dim
        else:
            self.ollama_encoder = None

        if self.use_time_gap_input:
            self.time_projection = nn.Sequential(
                nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU()
            )
            input_dim += 8
        else:
            self.time_projection = None

        self.visit_film = (
            VisitReliabilityFiLM(graph_emb_dim, meta_emb_dim,
                                 use_reliability=self.use_clinical_reliability,
                                 tau_days=reliability_tau_days, reliability_mode=clinical_reliability_mode,
                                 condition_mode=clinical_film_mode)
            if self.use_visit_film else None
        )

        d_model = hidden_dim * 2
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.attn_block = TemporalAttentionBlock(
            d_model, n_heads=n_attn_heads, dropout=dropout
        )
        self.pool_gate = nn.Linear(d_model, 1, bias=False)

        if self.use_metadata_branch:
            self.metadata_fusion = build_metadata_fusion(
                self.metadata_fusion_operator, graph_dim=d_model,
                metadata_dim=metadata_branch_dim, dropout=metadata_branch_dropout,
            )
        else:
            self.metadata_fusion = None

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self._last_hidden = None
        self._return_hidden = False
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.pool_gate.weight)
        # Initial equivalence to reference: visit-level FiLM is identity at start.
        if self.visit_film is not None:
            nn.init.zeros_(self.visit_film.gamma.weight)
            nn.init.zeros_(self.visit_film.gamma.bias)
            nn.init.zeros_(self.visit_film.beta.weight)
            nn.init.zeros_(self.visit_film.beta.bias)

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
        meta_age_days_seq: Optional[torch.Tensor] = None,
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
            if meta_num_seq is None:
                raise ValueError(
                    "Structured metadata is enabled but metadata tensors were not provided"
                )
            metadata_values = meta_num_seq
            if self.metadata_imputer is not None:
                compute_reconstruction = self.training and self.metadata_imputer_mode in {"joint", "scratch_joint"}
                imputed = self.metadata_imputer(
                    meta_num_seq, meta_mask_seq, lengths,
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
                baseline_any = meta_baseline_mask_seq.any(dim=-1, keepdim=True).to(metadata_values.dtype)
                encoder_present = baseline_any if encoder_present is None else torch.maximum(encoder_present.float(), baseline_any)
            metadata_visits = self.meta_encoder(
                metadata_values, meta_mask_seq, meta_cat_seq, encoder_present,
            )
            if self.baseline_encoder is not None:
                baseline_values = meta_baseline_num_seq if meta_baseline_num_seq is not None else torch.zeros_like(meta_num_seq)
                baseline_masks = meta_baseline_mask_seq if meta_baseline_mask_seq is not None else torch.zeros_like(meta_mask_seq)
                baseline_visits = self.baseline_encoder(
                    baseline_values, baseline_masks, meta_months_since_baseline_seq
                )
                metadata_visits = self.metadata_combiner(
                    torch.cat([metadata_visits, baseline_visits], dim=-1)
                )
                baseline_present = baseline_masks.any(dim=-1, keepdim=True).to(metadata_visits.dtype)
                if meta_present_seq is None:
                    metadata_temporal_present = baseline_present
                else:
                    metadata_temporal_present = torch.maximum(meta_present_seq.float(), baseline_present)
                metadata_visits = metadata_visits * metadata_temporal_present
            if self.visit_film is not None:
                pieces[0] = self.visit_film(
                    graph_seq, metadata_visits, meta_mask_seq,
                    metadata_temporal_present, meta_age_days_seq,
                )
            if self.use_metadata:
                pieces.append(metadata_visits)

        if self.use_ollama_metadata:
            if ollama_seq is None:
                raise ValueError(
                    "use_ollama_metadata=True but ollama_seq was not provided"
                )
            pieces.append(self.ollama_encoder(ollama_seq, ollama_present_seq))

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

        fused_visits = torch.cat(pieces, dim=-1)
        x = self.input_proj(fused_visits)
        positions = torch.arange(max_visits, device=graph_seq.device).unsqueeze(0)
        pad_mask = positions >= lengths.to(graph_seq.device).unsqueeze(1)
        x = self.attn_block(x, key_padding_mask=pad_mask)

        gate = self.pool_gate(x).squeeze(-1).masked_fill(pad_mask, float("-inf"))
        weights = F.softmax(gate, dim=-1).unsqueeze(-1)
        final_hidden = (x * weights).sum(dim=1)

        if self.use_metadata_branch:
            assert metadata_visits is not None
            metadata_hidden = self.metadata_branch(metadata_visits, lengths, metadata_temporal_present)
            final_hidden = self.metadata_fusion(final_hidden, metadata_hidden)

        self._last_hidden = final_hidden
        if self._return_hidden:
            return final_hidden
        return self.classifier(final_hidden)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.metadata_imputer is not None and self.metadata_imputer_mode == "frozen":
            self.metadata_imputer.eval()
        return self

    def set_return_hidden(self, flag: bool = True):
        self._return_hidden = bool(flag)

    @property
    def hidden_dim(self):
        return self.classifier[0].in_features

    def extra_repr(self) -> str:
        d_model = self.attn_block.norm1.normalized_shape[0]
        return (
            f"d_model={d_model}, n_heads={self.attn_block.attn.num_heads}, "
            f"metadata_branch={self.use_metadata_branch}"
        )
