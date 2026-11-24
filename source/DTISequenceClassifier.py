#!/usr/bin/env python3
"""Temporal classifier for DTI STGNN with optional previous-DX and metadata inputs."""
from __future__ import annotations

from typing import List, Optional
import torch
import torch.nn as nn

from metadata_imputer import BaselineAwareMaskedTransformerImputer, BaselineAwareGRUImputer, load_imputer_checkpoint


class VisitMetadataEncoder(nn.Module):
    """Simple MLP metadata encoder.

    Input is per-visit metadata:
      meta_num/mask: [B,T,N]
      meta_cat:      [B,T,C]
      meta_present:  [B,T,1]
    Output:
      [B,T,out_dim]
    """
    def __init__(
        self,
        num_dim: int,
        cat_cardinalities: List[int],
        out_dim: int = 64,
        cat_emb_dim: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_dim = int(num_dim)
        self.cat_cardinalities = list(cat_cardinalities)
        self.out_dim = int(out_dim)
        self.cat_emb_dim = int(cat_emb_dim)
        self.embs = nn.ModuleList([nn.Embedding(c, cat_emb_dim) for c in self.cat_cardinalities])
        in_dim = self.num_dim * 2 + len(self.cat_cardinalities) * cat_emb_dim + 1
        if in_dim == 0:
            self.net = None
        else:
            hidden = max(out_dim, 32)
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim),
                nn.ReLU(),
            )

    def forward(self, meta_num, meta_mask, meta_cat, meta_present):
        pieces = []
        if self.num_dim > 0:
            pieces.extend([meta_num, meta_mask])
        if len(self.embs) > 0:
            cat_pieces = []
            for j, emb in enumerate(self.embs):
                vals = meta_cat[..., j].clamp(min=0, max=emb.num_embeddings - 1)
                cat_pieces.append(emb(vals))
            pieces.append(torch.cat(cat_pieces, dim=-1))
        pieces.append(meta_present.float())
        x = torch.cat(pieces, dim=-1)
        return self.net(x)


class VisitMetadataTransformerEncoder(nn.Module):
    """FT-Transformer-style visit metadata encoder.

    Each metadata field is represented as a token. Numerical fields use the value,
    a learned field embedding, and a learned present/missing embedding. Categorical
    fields use a category embedding plus a learned field embedding. A CLS token is
    read out and projected to meta_emb_dim. The transformer is applied to each
    visit independently, then reshaped back to [B,T,out_dim].
    """
    def __init__(
        self,
        num_dim: int,
        cat_cardinalities: List[int],
        out_dim: int = 64,
        token_dim: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_dim = int(num_dim)
        self.cat_cardinalities = list(cat_cardinalities)
        self.out_dim = int(out_dim)
        self.token_dim = int(token_dim)
        self.n_cat = len(self.cat_cardinalities)
        self.n_tokens = self.num_dim + self.n_cat

        if self.n_tokens == 0:
            self.null = nn.Parameter(torch.zeros(out_dim))
            return

        self.cls = nn.Parameter(torch.zeros(1, 1, token_dim))

        if self.num_dim > 0:
            self.num_value_proj = nn.Linear(1, token_dim)
            self.num_field_emb = nn.Embedding(self.num_dim, token_dim)
            self.num_missing_emb = nn.Embedding(2, token_dim)
        else:
            self.num_value_proj = None
            self.num_field_emb = None
            self.num_missing_emb = None

        if self.n_cat > 0:
            self.cat_embs = nn.ModuleList([nn.Embedding(c, token_dim) for c in self.cat_cardinalities])
            self.cat_field_emb = nn.Embedding(self.n_cat, token_dim)
        else:
            self.cat_embs = nn.ModuleList()
            self.cat_field_emb = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(token_dim)
        self.out = nn.Sequential(
            nn.Linear(token_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, meta_num, meta_mask, meta_cat, meta_present):
        B, T = meta_present.shape[:2]
        if self.n_tokens == 0:
            return self.null.view(1, 1, -1).expand(B, T, -1)

        tokens = []
        BT = B * T

        if self.num_dim > 0:
            vals = meta_num.reshape(BT, self.num_dim, 1)
            mask = meta_mask.reshape(BT, self.num_dim).long().clamp(0, 1)
            field_ids = torch.arange(self.num_dim, device=meta_num.device).view(1, self.num_dim)
            num_tok = self.num_value_proj(vals)
            num_tok = num_tok + self.num_field_emb(field_ids) + self.num_missing_emb(mask)
            tokens.append(num_tok)

        if self.n_cat > 0:
            cats = meta_cat.reshape(BT, self.n_cat)
            cat_tokens = []
            for j, emb in enumerate(self.cat_embs):
                vals = cats[:, j].clamp(min=0, max=emb.num_embeddings - 1)
                field_id = torch.full((BT,), j, dtype=torch.long, device=meta_cat.device)
                cat_tokens.append(emb(vals) + self.cat_field_emb(field_id))
            tokens.append(torch.stack(cat_tokens, dim=1))

        x = torch.cat(tokens, dim=1)
        cls = self.cls.expand(BT, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        z = self.out(self.norm(x[:, 0]))

        # Zero out metadata embedding for visits where the metadata row was not present.
        present = meta_present.reshape(BT, 1).float()
        z = z * present
        return z.reshape(B, T, self.out_dim)


class OllamaEmbeddingProjector(nn.Module):
    """Trainable projection for frozen Ollama visit embeddings.

    Input:
      ollama_seq:         [B,T,E]
      ollama_present_seq: [B,T,1]
    Output:
      projected embedding [B,T,out_dim]

    The Ollama embedding model itself is frozen/cached. This module is trained
    inside the DTI-STGNN model.
    """
    def __init__(self, input_dim: int, out_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.input_dim = int(input_dim)
        self.out_dim = int(out_dim)
        if self.input_dim <= 0:
            self.null = nn.Parameter(torch.zeros(out_dim))
            self.net = None
        else:
            hidden = max(out_dim * 2, 128)
            self.net = nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

    def forward(self, ollama_seq, ollama_present_seq=None):
        B, T = ollama_seq.shape[:2]
        if self.net is None:
            return self.null.view(1, 1, -1).expand(B, T, -1)
        z = self.net(ollama_seq.float())
        if ollama_present_seq is not None:
            z = z * ollama_present_seq.float()
        return z




class BaselineMetadataEncoder(nn.Module):
    """Encode static baseline measurements plus elapsed time."""
    def __init__(self, num_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        hidden = max(out_dim * 2, 64)
        self.net = nn.Sequential(
            nn.Linear(num_dim * 2 + 1, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
        )

    def forward(self, values, masks, months_since_baseline):
        if months_since_baseline is None:
            months_since_baseline = torch.zeros((*values.shape[:2], 1), device=values.device, dtype=values.dtype)
        time = torch.log1p(torch.clamp(months_since_baseline.float(), min=0.0)) / 5.0
        encoded = self.net(torch.cat([values, masks, time], dim=-1))
        present = masks.any(dim=-1, keepdim=True).to(encoded.dtype)
        return encoded * present


class AddMetadataFusion(nn.Module):
    def __init__(self, graph_dim: int, metadata_dim: int, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(metadata_dim, graph_dim), nn.GELU(), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(graph_dim)
    def forward(self, graph_hidden, metadata_hidden):
        return self.norm(graph_hidden + self.proj(metadata_hidden))


class ConcatMetadataFusion(nn.Module):
    def __init__(self, graph_dim: int, metadata_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(graph_dim + metadata_dim, graph_dim),
            nn.LayerNorm(graph_dim), nn.GELU(), nn.Dropout(dropout),
        )
    def forward(self, graph_hidden, metadata_hidden):
        return self.net(torch.cat([graph_hidden, metadata_hidden], dim=-1))


class FiLMMetadataFusion(nn.Module):
    def __init__(self, graph_dim: int, metadata_dim: int, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(graph_dim)
        self.film = nn.Sequential(
            nn.Linear(metadata_dim, graph_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(graph_dim * 2, graph_dim * 2),
        )
    def forward(self, graph_hidden, metadata_hidden):
        gamma, beta = self.film(metadata_hidden).chunk(2, dim=-1)
        return self.norm(graph_hidden) * (1.0 + 0.5 * torch.tanh(gamma)) + beta


def build_metadata_fusion(operator: str, graph_dim: int, metadata_dim: int, dropout: float):
    op = str(operator).lower()
    if op == "gated":
        return GatedMetadataFusion(graph_dim, metadata_dim, dropout)
    if op == "add":
        return AddMetadataFusion(graph_dim, metadata_dim, dropout)
    if op == "concat":
        return ConcatMetadataFusion(graph_dim, metadata_dim, dropout)
    if op == "film":
        return FiLMMetadataFusion(graph_dim, metadata_dim, dropout)
    raise ValueError(f"Unknown metadata fusion operator: {operator}")


class MetadataTemporalBranch(nn.Module):
    """Independent temporal branch over per-visit structured metadata.

    The visit-level metadata encoder is shared with optional early fusion. This
    branch then applies a small Transformer over visits and learned masked soft
    pooling to produce one metadata representation per forecasting window.
    """

    def __init__(
        self,
        input_dim: int,
        branch_dim: int = 128,
        n_layers: int = 1,
        n_heads: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if branch_dim % n_heads != 0:
            raise ValueError(
                f"metadata_branch_dim ({branch_dim}) must be divisible by "
                f"metadata_branch_heads ({n_heads})"
            )
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=branch_dim,
            nhead=n_heads,
            dim_feedforward=branch_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pool_gate = nn.Linear(branch_dim, 1, bias=False)
        self.norm = nn.LayerNorm(branch_dim)
        nn.init.zeros_(self.pool_gate.weight)

    def forward(
        self,
        visit_embeddings: torch.Tensor,
        lengths: torch.Tensor,
        present: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, max_visits, _ = visit_embeddings.shape
        x = self.input_proj(visit_embeddings)
        positions = torch.arange(max_visits, device=x.device).unsqueeze(0)
        mask = positions >= lengths.to(x.device).unsqueeze(1)
        if present is not None:
            present_mask = present.squeeze(-1).to(x.device) <= 0
            mask = mask | present_mask

        # Transformer attention cannot accept a row with every position masked.
        # Temporarily unmask position 0, then force the final branch output to zero.
        all_missing = mask.all(dim=1)
        safe_mask = mask.clone()
        safe_mask[all_missing, 0] = False
        x = self.encoder(x, src_key_padding_mask=safe_mask)
        scores = self.pool_gate(x).squeeze(-1).masked_fill(safe_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (x * weights).sum(dim=1)
        pooled = self.norm(pooled)
        return pooled * (~all_missing).to(pooled.dtype).unsqueeze(1)


class GatedMetadataFusion(nn.Module):
    """Fuse an independent metadata branch into the graph-temporal hidden state."""

    def __init__(
        self,
        graph_dim: int,
        metadata_dim: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.meta_proj = nn.Sequential(
            nn.Linear(metadata_dim, graph_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(graph_dim * 2, graph_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(graph_dim)

    def forward(
        self,
        graph_hidden: torch.Tensor,
        metadata_hidden: torch.Tensor,
    ) -> torch.Tensor:
        meta = self.meta_proj(metadata_hidden)
        gate = self.gate(torch.cat([graph_hidden, meta], dim=-1))
        return self.norm(graph_hidden + gate * meta)


class DTISequenceClassifier(nn.Module):
    def __init__(
        self,
        graph_emb_dim: int = 512,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
        num_classes: int = 3,
        rnn_type: str = "LSTM",
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
    ):
        super().__init__()
        self.graph_emb_dim = int(graph_emb_dim)
        self.use_prev_dx = bool(use_prev_dx)
        self.use_metadata = bool(use_metadata)
        self.use_metadata_branch = bool(use_metadata_branch)
        self.use_ollama_metadata = bool(use_ollama_metadata)
        self.use_time_gap_input = bool(use_time_gap_input)
        self.rnn_type = str(rnn_type).upper()
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
                raise ValueError(f"Unknown meta_encoder_type: {meta_encoder_type}")
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
                nn.Linear(1, 8),
                nn.ReLU(),
                nn.Linear(8, 8),
                nn.ReLU(),
            )
            input_dim += 8
        else:
            self.time_projection = None

        if self.rnn_type not in {"LSTM", "GRU", "RNN"}:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")
        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}[self.rnn_type]
        common = dict(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        if self.rnn_type == "RNN":
            self.rnn = rnn_cls(**common, nonlinearity="tanh")
        else:
            self.rnn = rnn_cls(**common)

        out_dim = hidden_dim * (2 if bidirectional else 1)
        if self.use_metadata_branch:
            self.metadata_fusion = build_metadata_fusion(
                self.metadata_fusion_operator, graph_dim=out_dim,
                metadata_dim=metadata_branch_dim, dropout=metadata_branch_dropout,
            )
        else:
            self.metadata_fusion = None

        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self._last_hidden = None
        self._return_hidden = False

    def forward(
        self,
        graph_seq,
        lengths,
        prev_dx_seq=None,
        meta_num_seq=None,
        meta_mask_seq=None,
        meta_cat_seq=None,
        meta_present_seq=None,
        meta_baseline_num_seq=None,
        meta_baseline_mask_seq=None,
        meta_months_since_baseline_seq=None,
        meta_months_since_previous_seq=None,
        ollama_seq=None,
        ollama_present_seq=None,
        time_gaps_raw=None,
    ):
        pieces = [graph_seq]
        batch_size, max_visits, _ = graph_seq.shape

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
            if self.use_metadata:
                pieces.append(metadata_visits)

        if self.use_ollama_metadata:
            if ollama_seq is None:
                raise ValueError(
                    "use_ollama_metadata=True but ollama_seq was not provided"
                )
            pieces.append(self.ollama_encoder(ollama_seq, ollama_present_seq))

        if self.use_time_gap_input:
            if time_gaps_raw is None:
                time_value = torch.zeros(batch_size, 1, device=graph_seq.device)
            else:
                time_value = time_gaps_raw.view(batch_size, 1)
            time_embedding = self.time_projection(time_value).unsqueeze(1).expand(
                batch_size, max_visits, -1
            )
            pieces.append(time_embedding)

        fused_visits = torch.cat(pieces, dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            fused_visits,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        if self.rnn_type == "LSTM":
            _output, (hidden, _cell) = self.rnn(packed)
        else:
            _output, hidden = self.rnn(packed)

        if self.rnn.bidirectional:
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            final_hidden = hidden[-1]

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
