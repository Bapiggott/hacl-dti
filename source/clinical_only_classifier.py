#!/usr/bin/env python3
"""Clinical-only temporal predictor for v13 stable/change training.

The graph_seq parameter is accepted for trainer ABI compatibility but its VALUES
never enter the model.  Predictions use visit-level clinical/biomarker vectors,
optional previous diagnosis (clinical context) and the known forecast horizon.
The model has no graph encoder or DTI attention path.  The existing trainer may
still compute graph embeddings before invoking this classifier; they are unused.
"""
from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn
from DTISequenceClassifier import (
    VisitMetadataEncoder, VisitMetadataTransformerEncoder,
    MetadataTemporalBranch,
)

class ClinicalOnlyTemporalClassifier(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = 2,
        dropout: float = 0.3,
        use_prev_dx: bool = False,
        dx_emb_dim: int = 8,
        use_time_gap_input: bool = False,
        meta_num_dim: int = 0,
        meta_cat_cardinalities: Optional[List[int]] = None,
        meta_emb_dim: int = 64,
        meta_encoder_type: str = "ft_transformer",
        meta_token_dim: int = 32,
        meta_transformer_layers: int = 2,
        meta_transformer_heads: int = 4,
        metadata_branch_dim: int = 128,
        metadata_branch_layers: int = 1,
        metadata_branch_heads: int = 2,
        metadata_branch_dropout: float = 0.3,
        metadata_include_baseline: bool = False,
        metadata_imputer_mode: str = "none",
        use_ollama_metadata: bool = False,
        **_unused,
    ):
        super().__init__()
        if not meta_num_dim and not meta_cat_cardinalities:
            raise ValueError('Clinical-only requires nonempty structured metadata fields')
        if metadata_imputer_mode != 'none':
            raise ValueError('Clinical-only presently requires metadata_imputer_mode=none; do not silently drop imputation')
        if metadata_include_baseline:
            raise ValueError('Clinical-only presently requires metadata_include_baseline=False; do not silently drop baseline features')
        if use_ollama_metadata:
            raise ValueError('Clinical-only does not currently support Ollama embeddings')
        if meta_encoder_type in {'ft_transformer', 'transformer'}:
            self.meta_encoder = VisitMetadataTransformerEncoder(
                meta_num_dim, meta_cat_cardinalities or [], out_dim=meta_emb_dim,
                token_dim=meta_token_dim, n_layers=meta_transformer_layers,
                n_heads=meta_transformer_heads, dropout=dropout,
            )
        elif meta_encoder_type == 'mlp':
            self.meta_encoder = VisitMetadataEncoder(
                meta_num_dim, meta_cat_cardinalities or [],
                out_dim=meta_emb_dim, dropout=dropout,
            )
        else:
            raise ValueError(f'Unsupported clinical-only encoder: {meta_encoder_type}')
        self.metadata_branch = MetadataTemporalBranch(
            input_dim=meta_emb_dim, branch_dim=metadata_branch_dim,
            n_layers=metadata_branch_layers, n_heads=metadata_branch_heads,
            dropout=metadata_branch_dropout,
        )
        self.use_prev_dx = bool(use_prev_dx)
        self.use_time_gap_input = bool(use_time_gap_input)
        self.dx_embedding = nn.Embedding(3, dx_emb_dim) if use_prev_dx else None
        self.time_projection = (nn.Sequential(nn.Linear(1,8),nn.ReLU(),nn.Linear(8,8),nn.ReLU())
                                if use_time_gap_input else None)
        final_dim = metadata_branch_dim + (dx_emb_dim if use_prev_dx else 0) + (8 if use_time_gap_input else 0)
        self.classifier = nn.Sequential(
            nn.Linear(final_dim,64), nn.ReLU(), nn.Dropout(dropout),nn.Linear(64,num_classes)
        )
        self._last_hidden = None
        self._return_hidden = False
        self._imputation_loss = torch.tensor(0.0)

    def forward(
        self, graph_seq: torch.Tensor, lengths: torch.Tensor,
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
        # graph_seq intentionally not used, even for shape/device; graph features
        # cannot influence clinical-only predictions or their gradients.
        if meta_num_seq is None or meta_mask_seq is None or meta_cat_seq is None:
            raise ValueError('Clinical-only requires structured visit metadata tensors')
        bsz = meta_num_seq.shape[0]
        if lengths.shape[0] != bsz:
            raise ValueError('metadata batch size and lengths disagree')
        if meta_present_seq is None:
            meta_present_seq = (meta_mask_seq.sum(dim=-1,keepdim=True) > 0).float()
        visit_embeddings = self.meta_encoder(meta_num_seq,meta_mask_seq,meta_cat_seq,meta_present_seq)
        hidden = self.metadata_branch(visit_embeddings,lengths,meta_present_seq)
        pieces = [hidden]
        if self.use_prev_dx:
            if prev_dx_seq is None:
                raise ValueError('Clinical-only previous diagnosis enabled but prev_dx_seq missing')
            end = lengths.long().clamp(min=1).to(prev_dx_seq.device)-1
            recent_dx = prev_dx_seq.gather(1,end.unsqueeze(1)).squeeze(1).long()
            pieces.append(self.dx_embedding(recent_dx))
        if self.use_time_gap_input:
            if time_gaps_raw is None:
                raise ValueError('Clinical-only time gap enabled but time_gaps_raw missing')
            pieces.append(self.time_projection(time_gaps_raw.reshape(bsz,1)))
        hidden = torch.cat(pieces,dim=-1)
        self._last_hidden = hidden
        self._imputation_loss = hidden.sum()*0.0
        if self._return_hidden:
            return hidden
        return self.classifier(hidden)

    def set_return_hidden(self, flag: bool = True):
        self._return_hidden = bool(flag)

    @property
    def hidden_dim(self):
        return self.classifier[0].in_features
