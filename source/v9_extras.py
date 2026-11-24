"""
v9_extras.py  —  New components for DTI-STGNN v10
==================================================
Drop-in additions for dti_main_stable_change_v9.py.

Import in v9 by adding (after the existing imports):
    try:
        from v9_extras import *
    except ImportError as e:
        print(f"[v9_extras] not found: {e}")

New argparse flags (see patch_v9_to_v10.py):
    --use_improved_gnn        ImprovedGraphSAGE (PairNorm + residual + JK-Max + stochastic depth)
    --stochastic_depth_p F    Probability of skipping GNN layer 2 (default 0.15)
    --feature_mask_p F        Probability of zeroing each input feature dim (default 0.08)
    --loss_type logit_adjusted Logit-adjusted loss (replaces weighted CE, better calibration)
    --logit_adj_tau F         Log-prior scaling factor (default 1.0)
    --use_multiscale_delta    Add Δt=2 on top of existing Δt=1 delta
    --use_attention_temporal  Replace BiGRU with DTISequenceClassifierV2 (attention + learned pooling)
    --attn_heads N            Number of attention heads for V2 (must divide hidden_dim*2, default 2)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv, global_max_pool, global_mean_pool
    from torch_geometric.utils import dropout_edge
    _HAS_PYGEOMETRIC = True
except ImportError:
    _HAS_PYGEOMETRIC = False


# ══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

class LogitAdjustedLoss(nn.Module):
    """
    Logit-Adjusted Loss (Menon et al., NeurIPS 2020).

    Why this is better than weighted CrossEntropyLoss for our calibration problem:
    Weighted CE SCALES the gradient of each class — this distorts the probability
    surface, causing our raw probabilities to predict 81% changed when the real
    rate is 8.4%.  Logit adjustment SHIFTS each class logit by tau*log(pi_c)
    before the softmax.  The shift is additive in log-space so:
      (a) relative ordering within a class is preserved
      (b) calibration is maintained (Theorem 1 in paper)
      (c) threshold search is more stable across folds

    Usage:
        # Compute pi_pos from the OVERSAMPLED training distribution,
        # not from the raw 8.4:1 distribution.  At 2:1 oversample:
        #   pi_pos ≈ n_changed / (n_changed + n_stable) ≈ 0.333
        criterion = LogitAdjustedLoss(pi_pos=0.333, tau=1.0)
    """

    def __init__(
        self,
        pi_pos: float,
        tau: float = 1.0,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        if not (0.0 < pi_pos < 1.0):
            raise ValueError(f"pi_pos must be in (0, 1), got {pi_pos}")
        pi_neg = 1.0 - pi_pos
        # Class 0 (stable) gets log(pi_neg); class 1 (changed) gets log(pi_pos)
        adj = torch.tensor([math.log(pi_neg + 1e-8), math.log(pi_pos + 1e-8)]) * tau
        self.register_buffer("adjustment", adj)
        self.label_smoothing = label_smoothing
        self.tau = tau
        self.pi_pos = pi_pos

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted = logits + self.adjustment.to(logits.device)
        return F.cross_entropy(adjusted, targets, label_smoothing=self.label_smoothing)


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVED GRAPH ENCODER
# ══════════════════════════════════════════════════════════════════════════════

class PairNorm(nn.Module):
    """
    PairNorm (Zhao & Akoglu, ICLR 2020).

    Standard BatchNorm/LayerNorm fail on graphs because they normalise across
    graphs in a batch, mixing information.  PairNorm normalises WITHIN each
    graph:  (1) centers node features (prevents all-same collapse), then
    (2) scales so the total node pairwise distance is constant.  This is the
    core fix for over-smoothing in shallow GNNs on small graphs (116 nodes).
    """

    def __init__(self, mode: str = "PN", scale: float = 1.0) -> None:
        super().__init__()
        self.mode = mode   # "PN" = center+scale; "PN-SI" = scale only
        self.scale = scale

    def forward(
        self,
        x: torch.Tensor,           # [N, D]
        batch: Optional[torch.Tensor] = None,  # [N] graph membership
    ) -> torch.Tensor:
        if batch is None:
            if self.mode == "PN":
                x = x - x.mean(dim=0, keepdim=True)
            rms = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            return self.scale * x / rms

        # Batched case — normalise each graph independently
        num_graphs = int(batch.max().item()) + 1
        parts = []
        for g in range(num_graphs):
            mask = batch == g
            xg = x[mask]
            if self.mode == "PN":
                xg = xg - xg.mean(dim=0, keepdim=True)
            rms = (1e-6 + xg.pow(2).sum(dim=1).mean()).sqrt()
            parts.append(self.scale * xg / rms)
        return torch.cat(parts, dim=0)


class ResidualSAGEBlock(nn.Module):
    """
    Pre-activation GraphSAGE layer:  PairNorm → SAGEConv → GELU → + residual

    Pre-activation order (norm then conv) means gradients flow directly
    through the residual branch to early layers.  The residual projection
    handles the in_dim → out_dim mismatch on the first block.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        if not _HAS_PYGEOMETRIC:
            raise ImportError(
                "torch_geometric is required for ResidualSAGEBlock. "
                "Install with: pip install torch_geometric"
            )
        self.norm = PairNorm(mode="PN")
        self.conv = SAGEConv(in_dim, out_dim, aggr="mean")
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        # Residual: project to out_dim if dims don't match
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm(x, batch)
        h = self.conv(h, edge_index)      # SAGEConv uses mean aggregation
        h = self.drop(h)
        h = self.act(h)
        return h + self.res_proj(x)       # residual skip


class ImprovedGraphSAGE(nn.Module):
    """
    2-layer Residual GraphSAGE — drop-in replacement for GraphNeuralNetwork.

    Improvements over vanilla GraphSAGE:
    ① PairNorm after each layer  — prevents over-smoothing on 116-node graphs
    ② Residual connections       — preserves 1-hop local DTI signal
    ③ JK-Max (element-wise max)  — parameter-free multi-scale (concat hurts, shown in ablation)
    ④ Stochastic depth           — randomly skip layer 2 during training (p=0.15)
                                   forces each layer to be independently useful
                                   this should push the training peak from ~ep237 later
    ⑤ Feature masking            — zero each input feature dim (FA/MD/RD/AD) with p=0.08
                                   forces robustness to any single DTI metric

    DropEdge is moved INSIDE here (away from the DataLoader).  Pass dropedge_p=0.0
    if you want to turn it off.

    Output:
        global_max_pool(h_jk) ⊕ global_mean_pool(h_jk)  →  [B, output_dim * 2]
        Sets self.output_graph_dim = output_dim * 2  (matches current 256-dim interface
        when gnn_output_dim=128, so the downstream BiGRU sees the same shape).
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.25,          # kept for API compatibility, unused here
        dropedge_p: float = 0.15,
        stochastic_depth_p: float = 0.15,
        feature_mask_p: float = 0.08,
        **kwargs,                        # absorb unused GraphNeuralNetwork kwargs
    ) -> None:
        super().__init__()
        if not _HAS_PYGEOMETRIC:
            raise ImportError("torch_geometric is required for ImprovedGraphSAGE.")

        self.dropedge_p = dropedge_p
        self.sd_p = stochastic_depth_p
        self.fm_p = feature_mask_p

        self.layer1 = ResidualSAGEBlock(input_dim,  hidden_dim, dropout=0.1)
        self.layer2 = ResidualSAGEBlock(hidden_dim, output_dim, dropout=0.1)

        # This attribute is read by v9's make_encoder:
        #   graph_emb_dim = getattr(encoder, "output_graph_dim", args.gnn_output_dim * 2)
        # global_max + global_mean → output_dim * 2 = 256 (same as before)
        self.output_graph_dim = output_dim * 2

    def _feature_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Zero entire feature DIMS (not nodes) with prob fm_p. Scales surviving dims."""
        if not self.training or self.fm_p <= 0:
            return x
        # Mask is per-feature-dimension, broadcast over all nodes
        keep = torch.bernoulli(
            torch.full((x.size(-1),), 1.0 - self.fm_p, device=x.device)
        )
        # Scale to preserve expected activation magnitude
        return x * keep / (1.0 - self.fm_p + 1e-6)

    def forward(self, x, edge_index, batch=None, time_features=None, **kwargs):
        """
        Matches DTIWindowDataLoader_stable_change calling convention:
            encoder(big_batch.x, big_batch.edge_index, big_batch.batch, time_features)

        edge_weight is not passed by the DataLoader, so DropEdge applies to
        unweighted edges (standard for GraphSAGE with tractography top-k edges).
        """
        # ① Feature masking on raw DTI inputs (FA/MD/RD/AD)
        x = self._feature_mask(x)

        # ② DropEdge (training only, no edge weights to update)
        if self.training and self.dropedge_p > 0:
            edge_index, _ = dropout_edge(
                edge_index, p=self.dropedge_p, force_undirected=True
            )

        # edge_weight=None since DataLoader doesn't pass it
        edge_weight = None

        # ③ Layer 1 (always active)
        h1 = self.layer1(x, edge_index, edge_weight, batch)

        # ④ Layer 2 with stochastic depth
        skip_layer2 = self.training and (torch.rand(1).item() < self.sd_p)
        h2 = h1 if skip_layer2 else self.layer2(h1, edge_index, edge_weight, batch)

        # ⑤ JK-Max: element-wise max across both layers (no dim change)
        h = torch.max(h1, h2)

        # Graph-level readout: max + mean pooling → concat → output_dim*2
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        return torch.cat([global_max_pool(h, batch),
                          global_mean_pool(h, batch)], dim=-1)  # [B, output_dim*2]


# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL DELTA AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_multiscale_temporal_delta(
    graph_seq: torch.Tensor,       # [B, T, D]
    lengths:   torch.Tensor,       # [B] valid visit counts
    alpha1: float = 0.3,           # weight for Δt=1 (same as existing default)
    alpha2: float = 0.15,          # weight for Δt=2 (new)
) -> torch.Tensor:
    """
    Adds both Δt=1 and Δt=2 temporal deltas to the graph embedding sequence.

    The existing compute_temporal_delta only uses Δt=1 (current - previous).
    Δt=2 (current - two-visits-ago) captures slower, more stable trends in
    white matter microstructure — potentially more informative for AD progression
    than single-step differences.

    Drop-in replacement for compute_temporal_delta in _full_forward().
    Set --use_multiscale_delta to use this instead.
    """
    if not isinstance(graph_seq, torch.Tensor) or graph_seq.dim() != 3:
        return graph_seq
    B, T, D = graph_seq.shape

    delta1 = torch.zeros_like(graph_seq)
    delta2 = torch.zeros_like(graph_seq)

    if T > 1:
        delta1[:, 1:] = graph_seq[:, 1:] - graph_seq[:, :-1]
    if T > 2:
        delta2[:, 2:] = graph_seq[:, 2:] - graph_seq[:, :-2]

    # Zero out padding positions
    for b in range(B):
        l = int(lengths[b].item())
        if l < T:
            delta1[b, l:] = 0.0
            delta2[b, l:] = 0.0

    return graph_seq + alpha1 * delta1 + alpha2 * delta2


# ═══════════════════════════════════════════════════════════════════════════
# Section 4 — Classifier factory (V1 BiGRU  vs  V2 Attention)
# ═══════════════════════════════════════════════════════════════════════════

def build_classifier_v10(args, graph_emb_dim: int, num_classes: int, dataset):
    """
    Factory that returns DTISequenceClassifierV2 (attention) or the legacy
    DTISequenceClassifier (BiGRU) depending on --use_attention_temporal.

    This is a drop-in for the DTISequenceClassifier(...).to(device) block
    in v9's fold loop.  Usage in dti_main_stable_change_v9.py:

        # at top of file, after existing imports:
        from v9_extras import build_classifier_v10

        # inside the fold loop, replacing the DTISequenceClassifier block:
        classifier = build_classifier_v10(
            args, graph_emb_dim, num_classes, dataset
        ).to(device)

    Works identically when --use_attention_temporal is absent (falls back to
    original BiGRU), so it is safe to add unconditionally.
    """
    use_metadata = bool(
        getattr(dataset, "meta_num_dim", 0) > 0
        or len(getattr(dataset, "meta_cat_cardinalities", []) or []) > 0
    )
    use_ollama = bool(getattr(dataset, "ollama_dim", 0) > 0)

    common_kwargs = dict(
        graph_emb_dim           = graph_emb_dim,
        hidden_dim              = args.lstm_hidden_dim,
        num_layers              = getattr(args, "lstm_num_layers", 2),
        dropout                 = args.temporal_dropout,
        bidirectional           = getattr(args, "lstm_bidirectional", True),
        num_classes             = num_classes,
        rnn_type                = args.model_type,
        use_prev_dx             = args.use_prev_dx,
        use_metadata            = use_metadata,
        meta_num_dim            = getattr(dataset, "meta_num_dim", 0),
        meta_cat_cardinalities  = getattr(dataset, "meta_cat_cardinalities", []) or [],
        meta_emb_dim            = getattr(args, "meta_emb_dim", 64),
        meta_encoder_type       = getattr(args, "meta_encoder", "mlp"),
        meta_token_dim          = getattr(args, "meta_token_dim", 32),
        meta_transformer_layers = getattr(args, "meta_transformer_layers", 2),
        meta_transformer_heads  = getattr(args, "meta_transformer_heads", 4),
        use_ollama_metadata     = use_ollama,
        ollama_input_dim        = getattr(dataset, "ollama_dim", 0),
        ollama_emb_dim          = getattr(args, "ollama_emb_dim", 64),
        use_time_gap_input      = args.use_time_gap_input,
    )

    if getattr(args, "use_attention_temporal", False):
        try:
            import os, sys as _sys
            _here = os.path.dirname(os.path.abspath(__file__))
            if _here not in _sys.path:
                _sys.path.insert(0, _here)
            from DTISequenceClassifierV2 import DTISequenceClassifierV2
            n_heads = getattr(args, "attn_heads", 2)
            print(f"[classifier] DTISequenceClassifierV2  "
                  f"(attention n_heads={n_heads}  d_model={args.lstm_hidden_dim * 2})")
            return DTISequenceClassifierV2(n_attn_heads=n_heads, **common_kwargs)
        except ImportError as e:
            print(f"[WARNING] --use_attention_temporal requested but "
                  f"DTISequenceClassifierV2.py not found ({e}). "
                  f"Falling back to BiGRU.")

    from DTISequenceClassifier import DTISequenceClassifier
    print(f"[classifier] DTISequenceClassifier  (BiGRU {args.model_type})")
    return DTISequenceClassifier(**common_kwargs)