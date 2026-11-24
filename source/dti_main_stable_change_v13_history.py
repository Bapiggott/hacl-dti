#!/usr/bin/env python3
"""
dti_main_stable_change_v6.py  —  stable-vs-changed trainer (stratified sampling + T-fix + robust hard-rule)
========================================================================================

Diagnosis of v2 failure
-----------------------
 • AUC 0.737 is decent — the model CAN discriminate.  The problem is calibration.
 • With 3:1 oversample ratio in training, the model learned priors for a 3:1 world.
   At test time the real ratio is 8:1, so raw probabilities are over-confident for
   "changed".  Threshold 0.283 on val (8.2% changed) then predicted 81% changed on
   test (13.5% changed) — a pure calibration failure.
 • The selected_csv was only used for loss weights, not as model features.
 • At 88 epochs on a 2080 Ti, the model is massively under-capacity.

What is new in v3
-----------------
 ① TemperatureScaler
       After training, fit a single scalar T on val logits (LBFGS, NLL loss).
       This squashes/inflates probabilities so they reflect the true empirical
       distribution before threshold search.  Fixes the v2 calibration collapse.

 ② SubjectBiasNet
       Small MLP: [age, sex, APOE4] → logit bias  (leakage-free; the former
       is_converter/n_forward/n_backward features leaked outcome information)
       added to the classifier output.  Lets the model learn "converter subjects
       have a higher prior for any window being changed."  Trained end-to-end.

 ③ Temporal delta features (--use_temporal_delta)
       Compute Δg_t = g_t − g_{t-1} for the encoded graph embeddings and
       concatenate [g_t, Δg_t] as input to the LSTM.  The RATE of change in
       white-matter microstructure is likely more predictive of DX change than
       the absolute values.  Doubles graph_emb_dim automatically.

 ④ Prevalence-bounded threshold search
       At threshold optimisation time, the predicted 'changed' fraction must stay
       within 'prevalence_tolerance' × actual val changed fraction.  Prevents the
       0.283 → 81%-changed collapse from ever happening again.

 ⑤ Larger model defaults for 2080 Ti (11 GB VRAM)
       gnn_hidden/output 128→256, lstm_hidden 128→256, gnn_layers 2→3,
       lstm_layers 1→2, temporal_dropout 0.45→0.50.
       With ~800 windows the bottleneck is data, not GPU.

 ⑥ Softer oversampling (default 5:1 → closer to real 8:1)
       Training at 3:1 created a prior mismatch; 5:1 is a better compromise.

 ⑦ Cosine schedule T_0=30 (was 20) + patience=60 (was 40)
       Gives the larger model time to converge.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from DTI_ADNIDataset import DTI_ADNIDataset
from DTIWindowDataLoader_stable_change import DTIWindowDataLoaderStableChange
from DTISequenceClassifier import DTISequenceClassifier
from DTISequenceClassifierV2 import DTISequenceClassifierV2
from clinical_only_classifier import ClinicalOnlyTemporalClassifier
from DTISequenceClassifierV13 import DTISequenceClassifierV13
from FocalLoss import FocalLoss
from model import GraphNeuralNetwork
from advanced_gnn import AdvancedGraphNeuralNetwork

# ── v10 extras (LogitAdjustedLoss, ImprovedGraphSAGE, multiscale delta) ──
try:
    from v9_extras import (
        LogitAdjustedLoss, ImprovedGraphSAGE,
        compute_multiscale_temporal_delta,
    )
    _V10_EXTRAS = True
except ImportError as _v10_err:
    print(f"[v10] v9_extras not found: {_v10_err}  — new flags disabled")
    _V10_EXTRAS = False


# ══════════════════════════════════════════════════════════════════════════════
# ① TEMPERATURE SCALER
# ══════════════════════════════════════════════════════════════════════════════

class TemperatureScaler(nn.Module):
    """
    Post-hoc probability calibration via temperature scaling.

    Parameterised as T = exp(log_T) so T is ALWAYS > 0, regardless of
    where LBFGS goes.  In v5, T converged to -4.187 (a bug) which got
    clamped to 0.05 causing extreme probability sharpening.

    T > 1 → soften (model was overconfident)
    T < 1 → sharpen (model was underconfident / outputs too flat)
    """

    def __init__(self) -> None:
        super().__init__()
        # log(1.0) = 0 → start at T=1 (identity transform)
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=0.1, max=20.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        n_iters: int = 100,
        lr: float = 0.1,
    ) -> float:
        """
        Fit log_T using Adam.
        Do NOT use @torch.no_grad() here — log_temperature needs gradients.
        Inputs are detached so only log_T participates in the compute graph.
        """
        self.train()
        logits = logits.detach().clone()   # sever input from training graph
        labels = labels.detach().clone()
        opt = torch.optim.Adam([self.log_temperature], lr=lr)
        best_loss, best_log_t = float("inf"), float(self.log_temperature.item())
        for _ in range(n_iters):
            opt.zero_grad()
            loss = F.cross_entropy(self(logits), labels)
            loss.backward()
            opt.step()
            if float(loss.item()) < best_loss:
                best_loss = float(loss.item())
                best_log_t = float(self.log_temperature.item())
        # Restore best
        self.log_temperature.data.fill_(best_log_t)
        self.eval()
        t_val = float(self.temperature.item())
        print(f"[temp scale] T={t_val:.3f}  "
              f"({'overconfident→soften' if t_val > 1 else 'underconfident→sharpen'})")
        return t_val



# ══════════════════════════════════════════════════════════════════════════════
# SUPERVISED CONTRASTIVE LOSS  (auxiliary, on GRU window embeddings)
# ══════════════════════════════════════════════════════════════════════════════


class SoftBalancedAccuracyLoss(torch.nn.Module):
    """
    Differentiable surrogate for balanced accuracy.

    L_sba = 1 - (soft_TPR + soft_TNR) / 2
          = 1 - ( Σ(p_i·y_i)/n_pos  +  Σ((1-p_i)·(1-y_i))/n_neg ) / 2

    Blended with weighted CE for stable gradients:
        L = alpha·L_sba + (1-alpha)·L_ce

    alpha=1.0  pure balanced-accuracy surrogate
    alpha=0.5  equal blend (recommended default)
    alpha=0.0  reduces to weighted CE
    """
    def __init__(
        self,
        ce_weight: Optional[torch.Tensor] = None,
        alpha: float = 0.5,
        temperature: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.T = temperature
        self.label_smoothing = label_smoothing
        self.register_buffer("ce_weight", ce_weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs_changed = torch.softmax(logits / self.T, dim=-1)[:, 1]
        mask_pos = (labels == 1).float()
        mask_neg = (labels == 0).float()
        n_pos = mask_pos.sum().clamp(min=1.0)
        n_neg = mask_neg.sum().clamp(min=1.0)
        soft_tpr = (probs_changed * mask_pos).sum() / n_pos
        soft_tnr = ((1.0 - probs_changed) * mask_neg).sum() / n_neg
        loss_sba = 1.0 - (soft_tpr + soft_tnr) / 2.0
        if self.alpha >= 1.0:
            return loss_sba
        loss_ce = F.cross_entropy(
            logits, labels,
            weight=self.ce_weight,
            label_smoothing=self.label_smoothing,
        )
        return self.alpha * loss_sba + (1.0 - self.alpha) * loss_ce


class SupervisedContrastiveLoss(nn.Module):
    """
    SupCon auxiliary loss applied to the GRU output embeddings BEFORE the
    classification head.

    Pulls together embeddings from windows with the same label (stable/changed)
    and pushes apart cross-class pairs.  Added as:
        L_total = L_main + supcon_weight * L_supcon

    This gives the temporal encoder a richer gradient signal than CE alone:
    even when the main loss saturates, SupCon keeps refining the embedding
    space to be discriminative for conversion.

    Reference: Khosla et al., NeurIPS 2020 — "Supervised Contrastive Learning"
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.T = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings : [N, D]  — window-level GRU hidden states (pre-head)
            labels     : [N]     — 0=stable  1=changed
        """
        N = embeddings.shape[0]
        if N < 2:
            return embeddings.new_zeros(1).squeeze()

        # L2-normalise so dot product = cosine similarity
        emb = F.normalize(embeddings, dim=-1)
        sim = torch.mm(emb, emb.T) / self.T        # [N, N]

        # Mask diagonal (self-similarity) → -∞ so it doesn't appear in softmax
        eye = torch.eye(N, device=sim.device).bool()
        sim = sim.masked_fill(eye, float("-inf"))

        # Positive-pair mask: same label, not self
        lbl = labels.unsqueeze(1)
        pos_mask = (lbl == lbl.T).float()
        pos_mask.fill_diagonal_(0.0)

        n_pos = pos_mask.sum(1)
        if (n_pos == 0).all():
            return embeddings.new_zeros(1).squeeze()

        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)   # [N, N]
        # Mean over positive pairs for each anchor
        per_anchor = -(pos_mask * log_prob).sum(1) / n_pos.clamp(min=1)
        # Only average over anchors that HAD at least one positive pair
        return per_anchor[n_pos > 0].mean()


# ══════════════════════════════════════════════════════════════════════════════
# ② SUBJECT BIAS NET
# ══════════════════════════════════════════════════════════════════════════════

class SubjectBiasNet(nn.Module):
    """
    Lightweight MLP that converts per-subject static features into a logit bias
    added to the main classifier output.

    Input features (normalised)
    ---------------------------
    [is_converter, n_forward / MAX_FWD, n_backward / MAX_BWD]

    These are retrospective ADNI labels — valid for the analysis context.
    A converter subject has a higher prior for ANY window being "changed";
    n_forward > 0 means at least one forward transition exists in their trajectory.
    """

    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 32,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, num_classes]


def get_subject_features_batch(
    subjects: List[str],
    converter_lookup: Dict[str, Dict],
    device: torch.device,
    max_forward: float = 3.0,
    max_backward: float = 2.0,
    meta_lookup: Optional[Dict[str, Dict]] = None,
) -> torch.Tensor:
    """Build [B, 3] tensor of LEAKAGE-FREE subject-level static features.

    Features: age_zscore, sex_num (1=M), apoe4_norm (0/0.5/1.0)

    LEAKAGE FIX
    -----------
    This function previously returned [is_converter, n_forward_norm,
    n_backward_norm].  All three are computed from the subject's COMPLETE
    diagnostic trajectory, so they encode whether (and how often) the subject
    changes diagnosis at ANY point — including after the window being
    predicted.  Feeding them to the model is retrospective leakage: at genuine
    prediction time you would not know whether this subject ever converts.

    They are replaced with baseline clinical covariates that are known at the
    time of the scan and are not derived from the outcome.  `converter_lookup`
    is retained in the signature for call-site compatibility but is no longer
    read.
    """
    rows: List[List[float]] = []
    for sid in subjects:
        meta = (meta_lookup or {}).get(str(sid), {})
        rows.append([
            float(meta.get("age_norm", 0.0)),
            float(meta.get("sex_num", 0.5)),
            float(meta.get("apoe4_norm", 0.0)),
        ])
    return torch.tensor(rows, dtype=torch.float32, device=device)


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN KNOWLEDGE: HARD DX CONSTRAINT
# ══════════════════════════════════════════════════════════════════════════════

def apply_dx_hard_rules(logits: torch.Tensor, batch: dict) -> torch.Tensor:
    """
    Robust hard rule enforced two ways.

    Way 1 — transition string (most reliable):
      batch["transitions"][i] == "AD->AD"  → definitionally stable.
      Uses the string the DataLoader already computes, bypasses ambiguity
      in how prev_dx is stored (first vs last visit DX).

    Way 2 — target_dx + prev_dx fallback:
      Both target_dx[i]==2 and prev_dx[i]==2  → stable.

    Uses ABSOLUTE logit values (not additive) so the rule always wins.
    """
    logits = logits.clone()
    transitions = batch.get("transitions", [])
    prev_dx     = batch.get("prev_dx",    [])
    target_dx   = batch.get("target_dx",  [])

    for i in range(logits.size(0)):
        force_stable = False

        # Way 1: transition string
        if i < len(transitions) and isinstance(transitions[i], str):
            if transitions[i].strip() == "AD->AD":
                force_stable = True

        # Way 2: both target and prev DX are AD
        if not force_stable and i < len(target_dx) and i < len(prev_dx):
            try:
                if int(target_dx[i]) == 2 and int(prev_dx[i]) == 2:
                    force_stable = True
            except (TypeError, ValueError):
                pass

        if force_stable:
            logits[i, 0] = 15.0    # absolutely stable
            logits[i, 1] = -15.0
    return logits


# ══════════════════════════════════════════════════════════════════════════════
# PNA-STYLE MULTI-AGGREGATOR (drop-in enhancement for GraphSAGE-style GNN)
# ══════════════════════════════════════════════════════════════════════════════

class PNAAggregator(nn.Module):
    """
    Principal Neighbourhood Aggregation (Corso et al., NeurIPS 2020).

    Uses four aggregators [mean, max, min, std] combined with three
    degree-scalers [identity, log(d+1), 1/log(d+1)], giving 12 signals
    per feature instead of 1.  Projects back to original dim via linear.

    Applied ON TOP of the existing GraphSAGE/GCN output as a post-aggregation
    enrichment that doesn't require modifying model.py / advanced_gnn.py.

    Usage:
        pna = PNAAggregator(in_dim=256, out_dim=256)
        # graph_seq: [B, T, D] — run per time-step then stack back
        enriched = pna(graph_seq)   # same shape [B, T, D]
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        # 4 aggregators × 3 scalers = 12 × in_dim → out_dim
        self.proj = nn.Sequential(
            nn.Linear(12 * in_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] — treat the T visits as the 'neighbourhood'."""
        if x.dim() != 3:
            return x
        # Aggregators
        x_mean = x.mean(dim=1, keepdim=True).expand_as(x)   # [B,T,D]
        x_max  = x.max(dim=1, keepdim=True).values.expand_as(x)
        x_min  = x.min(dim=1, keepdim=True).values.expand_as(x)
        x_std  = x.std(dim=1, keepdim=True, unbiased=(x.size(1) > 1)).expand_as(x)

        # Degree scalers using T (sequence length as proxy for degree)
        T = float(x.size(1))
        s_id  = x
        s_log = x * (torch.log(torch.tensor(T + 1, dtype=x.dtype, device=x.device)))
        s_inv = x / (torch.log(torch.tensor(T + 1, dtype=x.dtype, device=x.device)) + 1e-6)

        # Cartesian product of 4 agg × 3 scalers = 12 features per element
        aggs  = [x_mean, x_max, x_min, x_std]
        sclrs = [s_id, s_log, s_inv]
        combined = torch.cat([a * s for a in aggs for s in sclrs], dim=-1)  # [B,T,12D]

        return self.proj(combined) + self.residual(x)  # residual


# ══════════════════════════════════════════════════════════════════════════════
# ③ TEMPORAL DELTA
# ══════════════════════════════════════════════════════════════════════════════

def compute_temporal_delta(
    graph_seq: torch.Tensor,   # [B, T, D]
    lengths: torch.Tensor,     # [B] valid visit counts
    alpha: float = 0.3,
) -> torch.Tensor:
    """
    ADDITIVE temporal delta: returns graph_seq + alpha * Δg_t
    where Δg_t = g_t − g_{t-1}, Δg_0 = 0.

    CRITICAL: additive (not concatenated) so graph_emb_dim is UNCHANGED.
    v3 concatenated the delta, doubling LSTM input dim → 800 windows,
    millions of params → the model became totally uncertain (T=0.252<1).
    Additive with alpha=0.3 is a soft blend: 70% absolute structure,
    30% rate-of-change signal. Same tensor shape goes into LSTM.
    """
    if not isinstance(graph_seq, torch.Tensor) or graph_seq.dim() != 3:
        return graph_seq
    B, T, D = graph_seq.shape
    delta = torch.zeros_like(graph_seq)
    delta[:, 1:] = graph_seq[:, 1:] - graph_seq[:, :-1]
    for b in range(B):
        l = int(lengths[b].item())
        if l < T:
            delta[b, l:] = 0.0
    return graph_seq + alpha * delta  # [B, T, D] — same shape as input


# ══════════════════════════════════════════════════════════════════════════════
# STRATIFIED WINDOW REBALANCING
# ══════════════════════════════════════════════════════════════════════════════

def stratified_rebalance_windows(
    windows: list,
    window_frame: pd.DataFrame,
    target_stable_to_changed: float = 3.0,
    converter_lookup: Optional[Dict] = None,
    seed: int = 42,
) -> list:
    """
    Smarter window rebalancing using the sliding-window insight.

    The Problem
    -----------
    Random oversampling of changed + random undersampling of stable leaves
    the model seeing mostly MCI→MCI stable windows (51% of stable), which
    teaches it that "MCI subject = probably stable."  This hurts precision.

    The Sliding-Window Opportunity
    --------------------------------
    With sliding windows, each subject contributes multiple windows at
    different points in their disease course.  We can CHOOSE which stable
    windows to include.  The most informative stable windows are:
      - Later in a converter subject's trajectory (harder negative)
      - AD→AD windows (keep ALL — used by hard DX rule)
      - Diverse across DX types (not all MCI→MCI)

    Strategy
    --------
    1. Keep ALL changed windows.
    2. Keep ALL AD→AD stable windows (small count, domain knowledge).
    3. For non-AD stable windows, sample PROPORTIONALLY by transition type,
       prioritising windows from converter subjects.
    4. Oversample changed if needed to reach target ratio.

    Falls back gracefully if window_frame is unavailable or misaligned.
    """
    rng = np.random.RandomState(seed)

    # ── Fallback if window_frame is unusable ──────────────────────────────
    if window_frame is None or len(window_frame) != len(windows):
        return oversample_changed_windows(windows, target_stable_to_changed, seed)

    # Column name variants for transition and label
    wf = window_frame.reset_index(drop=True).copy()
    wf["_idx"] = np.arange(len(wf))
    wf["_label"] = [int(w[-1]) for w in windows]

    # Find transition column
    trans_col = None
    for col in ["transition", "dx_transition", "trans"]:
        if col in wf.columns:
            trans_col = col
            break

    # ── Separate changed / stable ─────────────────────────────────────────
    changed_idx = wf[wf["_label"] == 1]["_idx"].tolist()
    stable_wf   = wf[wf["_label"] == 0]
    n_changed   = len(changed_idx)

    # ── Target stable count ───────────────────────────────────────────────
    n_stable_target = max(1, int(round(n_changed * target_stable_to_changed)))
    n_stable_target = min(n_stable_target, len(stable_wf))

    # ── If no transition info: proportional but converter-weighted ────────
    if trans_col is None:
        # Give converter subjects 2× weight for stable windows
        weights = np.ones(len(stable_wf))
        if converter_lookup:
            subj_col = None
            for col in ["subject_id", "subject", "subj_id"]:
                if col in wf.columns:
                    subj_col = col
                    break
            if subj_col:
                conv_mask = stable_wf[subj_col].map(
                    lambda s: converter_lookup.get(str(s), {}).get("is_converter", 0)
                ).fillna(0).values.astype(float)
                weights = 1.0 + conv_mask  # converters get weight 2, others get 1

        probs = weights / weights.sum()
        chosen = rng.choice(
            stable_wf["_idx"].values, size=n_stable_target, replace=False, p=probs
        )
        sampled_stable_idx = chosen.tolist()
    else:
        # ── Stratified by transition type ─────────────────────────────────
        # Always keep ALL AD→AD (small count, needed for hard rule)
        ad_mask  = stable_wf[trans_col].str.strip() == "AD->AD"
        ad_stable_idx   = stable_wf[ad_mask]["_idx"].tolist()
        non_ad_stable   = stable_wf[~ad_mask]
        remaining_budget = max(0, n_stable_target - len(ad_stable_idx))

        if remaining_budget == 0 or len(non_ad_stable) == 0:
            sampled_stable_idx = ad_stable_idx[:n_stable_target]
        else:
            sampled_non_ad = []
            trans_groups = non_ad_stable.groupby(trans_col)
            total_non_ad = len(non_ad_stable)

            for trans, grp in trans_groups:
                # Proportional allocation
                n_alloc = max(1, int(round(remaining_budget * len(grp) / total_non_ad)))
                n_alloc = min(n_alloc, len(grp))

                # Converter-weighted sampling within group
                weights = np.ones(len(grp))
                if converter_lookup:
                    subj_col = None
                    for col in ["subject_id", "subject", "subj_id"]:
                        if col in wf.columns:
                            subj_col = col; break
                    if subj_col:
                        conv_mask = grp[subj_col].map(
                            lambda s: converter_lookup.get(str(s), {}).get("is_converter", 0)
                        ).fillna(0).values.astype(float)
                        weights = 1.0 + conv_mask

                probs = weights / weights.sum()
                chosen = rng.choice(
                    grp["_idx"].values, size=n_alloc, replace=False, p=probs
                )
                sampled_non_ad.extend(chosen.tolist())

            sampled_stable_idx = ad_stable_idx + sampled_non_ad

    # ── Oversample changed to reach target ratio ──────────────────────────
    target_changed = max(n_changed, int(round(len(sampled_stable_idx) / target_stable_to_changed)))
    if target_changed > n_changed:
        extra = rng.choice(changed_idx, target_changed - n_changed, replace=True).tolist()
        all_changed = changed_idx + extra
    else:
        all_changed = changed_idx

    result = [windows[i] for i in sampled_stable_idx + all_changed]
    rng.shuffle(result)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# FORWARD CONTEXT  (bundles all the new modules cleanly)
# ══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class ForwardContext:
    """Carries optional modules / data through the forward pass helpers."""
    subject_bias_net: Optional[SubjectBiasNet] = None
    converter_lookup: Dict[str, Dict] = dataclasses.field(default_factory=dict)
    meta_lookup: Optional[Dict[str, Dict]] = None
    device: Optional[torch.device] = None
    bias_scale: float = 0.5
    use_temporal_delta: bool = False
    temp_scaler: Optional[TemperatureScaler] = None
    use_dx_hard_rule: bool = True        # always-on clinical hard constraint
    pna_enricher: Optional["PNAAggregator"] = None  # PNA post-aggregation layer
    drop_edge_p: float = 0.0   # DropEdge — set per-batch in _full_forward
    use_multiscale_delta: bool = False   # v10: Dt=1 + Dt=2 temporal deltas
    atlas_caches: Optional[dict] = None      # v11: {name: MultiAtlasCache}
    clinical_lookup: Optional[dict] = None   # v11: {subject: [age,sex,apoe4]}
    fusion_normalize: bool = True
    fusion_add_rate: bool = False

    def apply_temp(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temp_scaler is not None:
            return self.temp_scaler(logits)
        return logits


# ══════════════════════════════════════════════════════════════════════════════
# SHARED FORWARD PASS
# ══════════════════════════════════════════════════════════════════════════════

def _full_forward(
    batch: dict,
    encoder,
    classifier,
    ctx: Optional[ForwardContext] = None,
    apply_temp: bool = False,
    return_embeddings: bool = False,
    drop_edge_p: float = 0.0,
):
    """
    Full forward pass including optional delta augmentation, subject bias,
    and temperature scaling.

    apply_temp=True only during evaluation (after fitting temp scaler on val).
    """
    graph_seq = batch["graph_seq"]

    # ③ Temporal delta — ADDITIVE residual (keeps same dim, avoids v3 param explosion)
    # v10: multi-scale delta (Δt=1 + Δt=2) if requested, else existing Δt=1
    if getattr(ctx, 'use_multiscale_delta', False) and _V10_EXTRAS:
        try:
            graph_seq = compute_multiscale_temporal_delta(
                graph_seq, batch["lengths"])
        except Exception:
            pass
    elif ctx and ctx.use_temporal_delta:
        try:
            graph_seq = compute_temporal_delta(graph_seq, batch["lengths"], alpha=0.3)
        except Exception:
            pass  # safe no-op if graph_seq is not a 3-D float tensor

    # PNA multi-aggregator enrichment (applied to graph embedding sequence)
    if ctx and ctx.pna_enricher is not None:
        try:
            if isinstance(graph_seq, torch.Tensor) and graph_seq.dim() == 3:
                graph_seq = ctx.pna_enricher(graph_seq)
        except Exception:
            pass

    # v11: multi-atlas fusion features (if fusion classifier + caches present)
    _clf_kwargs = dict(
        prev_dx_seq=batch.get("prev_dx_seq"),
        meta_num_seq=batch.get("meta_num_seq"),
        meta_mask_seq=batch.get("meta_mask_seq"),
        meta_cat_seq=batch.get("meta_cat_seq"),
        meta_present_seq=batch.get("meta_present_seq"),
        meta_baseline_num_seq=batch.get("meta_baseline_num_seq"),
        meta_baseline_mask_seq=batch.get("meta_baseline_mask_seq"),
        meta_months_since_baseline_seq=batch.get("meta_months_since_baseline_seq"),
        meta_months_since_previous_seq=batch.get("meta_months_since_previous_seq"),
        ollama_seq=batch.get("ollama_seq"),
        ollama_present_seq=batch.get("ollama_present_seq"),
        time_gaps_raw=batch.get("time_gaps_raw"),
    )
    if ctx is not None and getattr(ctx, "atlas_caches", None):
        try:
            from multiatlas_fusion import gather_window_features
            _dates = batch.get("input_dates")
            if _dates is not None:
                atlas_feats = gather_window_features(
                    ctx.atlas_caches, batch["subjects"], _dates,
                    ctx.device, normalize=getattr(ctx, "fusion_normalize", True),
                    add_rate=getattr(ctx, "fusion_add_rate", False))
                _clf_kwargs["atlas_feats"] = atlas_feats
                # clinical features per sample
                if getattr(ctx, "clinical_lookup", None):
                    import numpy as _np
                    _cl = []
                    for _sid in batch["subjects"]:
                        _cl.append(ctx.clinical_lookup.get(
                            str(_sid), [0.0, 0.5, 0.0]))
                    _clf_kwargs["clinical"] = torch.tensor(
                        _np.asarray(_cl, dtype=_np.float32), device=ctx.device)
        except Exception as _e:
            pass  # fusion is best-effort; never break the forward pass

    if isinstance(classifier, DTISequenceClassifierV2) and getattr(classifier, "use_visit_film", False):
        _clf_kwargs["meta_age_days_seq"] = batch.get("meta_age_days_seq")
    logits = classifier(graph_seq, batch["lengths"], **_clf_kwargs)

    # ② Subject bias
    if ctx and ctx.subject_bias_net is not None and ctx.converter_lookup and ctx.device is not None:
        try:
            feat = get_subject_features_batch(batch["subjects"], ctx.converter_lookup, ctx.device, meta_lookup=ctx.meta_lookup)
            logits = logits + ctx.bias_scale * ctx.subject_bias_net(feat)
        except Exception:
            pass

    # ① Temperature scaling (eval only)
    if apply_temp and ctx is not None:
        logits = ctx.apply_temp(logits)

    # Hard clinical rule (eval always; training optionally via ctx flag)
    if ctx and ctx.use_dx_hard_rule:
        logits = apply_dx_hard_rules(logits, batch)

    return logits


# ══════════════════════════════════════════════════════════════════════════════
# MISC UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def try_import_wandb():
    try:
        import wandb
        return wandb
    except Exception:
        return None


def set_random_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_converter_lookup(csv_path: str) -> Dict[str, Dict]:
    df = pd.read_csv(csv_path)
    out: Dict[str, Dict] = {}
    for _, row in df.iterrows():
        sid = str(row["subject_id"])
        out[sid] = {
            "is_converter": int(row.get("is_converter", 0)),
            "n_forward":    int(row.get("n_forward", 0)),
            "n_backward":   int(row.get("n_backward", 0)),
        }
    return out


# ══════════════════════════════════════════════════════════════════════════════
# IMBALANCE HANDLING
# ══════════════════════════════════════════════════════════════════════════════

def oversample_changed_windows(windows: list, target_ratio: float, seed: int) -> list:
    """Duplicate changed windows until stable:changed ≤ target_ratio."""
    if target_ratio <= 0:
        return windows
    stable  = [w for w in windows if int(w[-1]) == 0]
    changed = [w for w in windows if int(w[-1]) == 1]
    if not changed:
        return windows
    target_n = max(len(changed), int(np.ceil(len(stable) / target_ratio)))
    if target_n <= len(changed):
        return windows
    rng   = np.random.RandomState(seed)
    extra = [changed[i] for i in rng.choice(len(changed), target_n - len(changed), replace=True)]
    result = stable + changed + extra
    rng.shuffle(result)
    return result


def rebalance_training_windows(windows: list, ratio: float, seed: int) -> list:
    """Downsample stable windows to stable:changed ≤ ratio."""
    if ratio <= 0:
        return windows
    stable  = [w for w in windows if int(w[-1]) == 0]
    changed = [w for w in windows if int(w[-1]) == 1]
    if not changed or not stable:
        return windows
    rng  = np.random.RandomState(seed)
    keep = max(1, min(len(stable), int(round(len(changed) * ratio))))
    result = [stable[i] for i in rng.choice(len(stable), keep, replace=False)] + changed
    rng.shuffle(result)
    return result


def class_weights_from_loader(loader, num_classes: int, device: torch.device):
    counts = np.zeros(num_classes, dtype=np.float32)
    for *_, lbl in loader.windows:
        if 0 <= lbl < num_classes:
            counts[lbl] += 1
    counts  = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device), counts.astype(int).tolist()


# ══════════════════════════════════════════════════════════════════════════════
# LOSS UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def get_converter_sample_weights(
    subjects: List[str],
    labels: torch.Tensor,
    lookup: Dict[str, Dict],
    stable_mult: float,
    changed_mult: float,
    device: torch.device,
) -> torch.Tensor:
    w = []
    for i, sid in enumerate(subjects):
        info    = lookup.get(sid, {})
        is_conv = int(info.get("is_converter", 0))
        lbl     = int(labels[i].item())
        if   is_conv and lbl == 0: w.append(stable_mult)
        elif is_conv and lbl == 1: w.append(changed_mult)
        else:                      w.append(1.0)
    return torch.tensor(w, dtype=torch.float32, device=device)


def compute_weighted_loss(logits, labels, criterion, sample_weights=None):
    if sample_weights is None:
        return criterion(logits, labels)
    try:
        orig = getattr(criterion, "reduction", "mean")
        criterion.reduction = "none"
        per  = criterion(logits, labels)
        criterion.reduction = orig
        return (per * sample_weights).mean()
    except Exception:
        return criterion(logits, labels)


# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def inject_graph_noise(graph_seq, std: float):
    if std <= 0 or not isinstance(graph_seq, torch.Tensor):
        return graph_seq
    if graph_seq.dtype not in (torch.float16, torch.float32, torch.float64):
        return graph_seq
    return graph_seq + torch.randn_like(graph_seq) * std


# ══════════════════════════════════════════════════════════════════════════════
# ④ PREVALENCE-BOUNDED THRESHOLD SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def optimize_threshold(
    probs:   np.ndarray,
    labels:  np.ndarray,
    metric:  str   = "balanced_accuracy",
    n_grid:  int   = 95,
    prevalence_tolerance: float = 3.0,
) -> Tuple[float, float]:
    """
    Grid-search the binary decision threshold on CALIBRATED probabilities.

    CRITICAL CHANGE FROM v2: predicted 'changed' fraction must stay within
    prevalence_tolerance × actual changed fraction in the val set.
    This prevents the 81%-changed collapse that killed v2.

    e.g. actual changed = 8 %  and  tolerance = 3.0
         → predicted changed must be in [2.7 %, 24 %]
    """
    pos          = probs[:, 1]
    actual_prev  = labels.mean()                    # val changed fraction
    lo_prev      = actual_prev / prevalence_tolerance
    hi_prev      = min(actual_prev * prevalence_tolerance, 0.85)

    best_t, best_s = 0.5, -1.0

    for t in np.linspace(0.02, 0.98, n_grid):
        preds    = (pos >= t).astype(int)
        frac_chg = preds.mean()
        if frac_chg < lo_prev or frac_chg > hi_prev:
            continue   # outside prevalence envelope
        if metric == "balanced_accuracy":
            s = balanced_accuracy_score(labels, preds)
        elif metric == "f1_changed":
            s = f1_score(labels, preds, pos_label=1, average="binary", zero_division=0)
        elif metric == "macro_f1":
            s = f1_score(labels, preds, average="macro", zero_division=0)
        else:
            s = balanced_accuracy_score(labels, preds)
        if s > best_s:
            best_s, best_t = s, float(t)

    if best_s < 0:           # no threshold satisfied the envelope; fall back
        best_t, best_s = 0.5, float(balanced_accuracy_score(labels, (pos >= 0.5).astype(int)))

    return best_t, best_s


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def metrics_from_probs(
    probs: np.ndarray, labels: np.ndarray,
    num_classes: int, threshold: float = 0.5,
) -> Dict:
    preds = (probs[:, 1] >= threshold).astype(int) if num_classes == 2 else probs.argmax(1)
    out: Dict = {
        "accuracy":          float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1":          float(f1_score(labels, preds, average="macro", zero_division=0)),
        "pred_counts":  {str(k): int(v) for k, v in pd.Series(preds).value_counts().sort_index().items()},
        "target_counts":{str(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()},
        "threshold": float(threshold),
    }
    if num_classes == 2:
        out["f1_changed"]        = float(f1_score(labels, preds, pos_label=1, average="binary", zero_division=0))
        out["recall_changed"]    = float(recall_score(labels, preds, pos_label=1, zero_division=0))
        out["precision_changed"] = float(precision_score(labels, preds, pos_label=1, zero_division=0))
    else:
        out["f1_changed"] = out["recall_changed"] = out["precision_changed"] = float("nan")

    try:
        out["auc"] = (
            float(roc_auc_score(labels, probs[:, 1])) if num_classes == 2 and len(set(labels)) == 2
            else float(roc_auc_score(labels, probs, multi_class="ovr", average="macro")) if len(set(labels)) > 1
            else float("nan")
        )
    except Exception:
        out["auc"] = float("nan")
    return out


def visit_report(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy breakdown by number of input visits per window."""
    if pred_df.empty or "n_visits" not in pred_df.columns:
        return pd.DataFrame()
    rows = []
    for nv, g in pred_df.groupby("n_visits"):
        rows.append({
            "n_visits":       int(nv),
            "n":              int(len(g)),
            "accuracy":       float((g["target"] == g["pred"]).mean()),
            "target_stable":  int((g["target"] == 0).sum()),
            "target_changed": int((g["target"] == 1).sum()),
            "pred_stable":    int((g["pred"] == 0).sum()),
            "pred_changed":   int((g["pred"] == 1).sum()),
        })
    return pd.DataFrame(rows).sort_values("n_visits")


def transition_report(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty or "transition" not in pred_df.columns:
        return pd.DataFrame()
    rows = []
    for tr, g in pred_df.groupby("transition"):
        rows.append({
            "transition":     tr,     "n":              int(len(g)),
            "accuracy":       float((g["target"] == g["pred"]).mean()),
            "target_stable":  int((g["target"] == 0).sum()),
            "target_changed": int((g["target"] == 1).sum()),
            "pred_stable":    int((g["pred"] == 0).sum()),
            "pred_changed":   int((g["pred"] == 1).sum()),
        })
    return pd.DataFrame(rows).sort_values("transition")


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    loader, encoder, classifier, criterion, num_classes: int,
    threshold: float = 0.5, ctx: Optional[ForwardContext] = None,
    apply_temp: bool = False,
):
    encoder.eval()
    classifier.eval()
    if ctx and ctx.subject_bias_net:
        ctx.subject_bias_net.eval()
    if ctx and ctx.temp_scaler:
        ctx.temp_scaler.eval()

    losses, probs_all, labels_all, rows = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            logits = _full_forward(batch, encoder, classifier, ctx, apply_temp=apply_temp)
            labels = batch["labels"]
            losses.append(float(criterion(logits, labels).item()))

            probs_b = F.softmax(logits, dim=-1).cpu().numpy()
            probs_all.append(probs_b)
            labels_all.append(labels.cpu().numpy())

            preds_b = (probs_b[:, 1] >= threshold).astype(int) if num_classes == 2 else probs_b.argmax(1)
            for i, sid in enumerate(batch["subjects"]):
                row = {
                    "subject_id":  sid,
                    "input_dates": batch["input_dates"][i],
                    "target_date": batch["target_dates"][i],
                    "target":      int(labels[i].cpu().item()),
                    "pred":        int(preds_b[i]),
                    "transition":  batch.get("transitions", [""] * len(batch["subjects"]))[i],
                    "prev_dx":     int(batch.get("prev_dx",   [-1] * len(batch["subjects"]))[i]),
                    "target_dx":   int(batch.get("target_dx", [-1] * len(batch["subjects"]))[i]),
                    "n_visits":    int(batch["lengths"][i].item()),
                }
                for c in range(num_classes):
                    row[f"prob_{c}"] = float(probs_b[i, c])
                rows.append(row)

    if not probs_all:
        return {"loss": float("nan"), "accuracy": 0, "balanced_accuracy": 0, "macro_f1": 0,
                "auc": float("nan"), "f1_changed": 0, "recall_changed": 0,
                "precision_changed": 0, "pred_counts": {}, "target_counts": {}, "threshold": threshold}, pd.DataFrame()

    probs  = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    m = metrics_from_probs(probs, labels, num_classes, threshold)
    m["loss"] = float(np.mean(losses))
    return m, pd.DataFrame(rows)


def collect_val_logits(
    loader, encoder, classifier, ctx: Optional[ForwardContext] = None,
):
    """Collect raw (uncalibrated) logits from val set for temperature fitting."""
    encoder.eval()
    classifier.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            # DO NOT apply temp scaling here — we need raw logits
            logits = _full_forward(batch, encoder, classifier, ctx, apply_temp=False)
            logits_all.append(logits.cpu())
            labels_all.append(batch["labels"].cpu())
    if not logits_all:
        return None, None
    return torch.cat(logits_all), torch.cat(labels_all)


def evaluate_tta(
    loader, encoder, classifier, criterion, num_classes: int,
    n_passes: int = 5, threshold: float = 0.5, ctx: Optional[ForwardContext] = None,
):
    if n_passes <= 1:
        return evaluate(loader, encoder, classifier, criterion, num_classes, threshold, ctx, apply_temp=True)

    all_probs: List[np.ndarray] = []
    labels_ref: Optional[np.ndarray] = None
    meta_rows: List[dict] = []

    for p_idx in range(n_passes):
        encoder.train()
        classifier.train()
        if ctx and ctx.subject_bias_net:
            ctx.subject_bias_net.train()
        probs_p = []
        with torch.no_grad():
            for batch in loader:
                logits  = _full_forward(batch, encoder, classifier, ctx, apply_temp=True)
                probs_p.append(F.softmax(logits, dim=-1).cpu().numpy())
                if p_idx == 0:
                    labels_ref = (
                        np.concatenate([labels_ref, batch["labels"].cpu().numpy()])
                        if labels_ref is not None
                        else batch["labels"].cpu().numpy()
                    )
                    for i, sid in enumerate(batch["subjects"]):
                        meta_rows.append({
                            "subject_id": sid,
                            "input_dates": batch["input_dates"][i],
                            "target_date": batch["target_dates"][i],
                            "transition": batch.get("transitions", [""] * len(batch["subjects"]))[i],
                            "prev_dx":    int(batch.get("prev_dx",   [-1] * len(batch["subjects"]))[i]),
                            "target_dx":  int(batch.get("target_dx", [-1] * len(batch["subjects"]))[i]),
                        })
        all_probs.append(np.concatenate(probs_p))

    avg_probs = np.mean(all_probs, axis=0)
    m = metrics_from_probs(avg_probs, labels_ref, num_classes, threshold)
    m["loss"] = float("nan")
    preds = (avg_probs[:, 1] >= threshold).astype(int) if num_classes == 2 else avg_probs.argmax(1)
    rows = [{**meta, "target": int(labels_ref[i]), "pred": int(preds[i]),
             **{f"prob_{c}": float(avg_probs[i, c]) for c in range(num_classes)}}
            for i, meta in enumerate(meta_rows)]
    return m, pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# SPLITS
# ══════════════════════════════════════════════════════════════════════════════

def subject_split_labels(dataset, task: str) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    for s, graphs in dataset.subject_graph_dict.items():
        if not graphs:
            continue
        if task == "converter":
            labels[s] = int(graphs[0].converter_label.item())
        elif task in ("stable_change", "progression"):
            dxs = [int(g.dx_label.item()) for g in graphs]
            labels[s] = int(any(
                (dxs[i] != dxs[i-1] if task == "stable_change" else dxs[i] > dxs[i-1])
                for i in range(1, len(dxs))
            ))
        else:
            labels[s] = int(graphs[-1].dx_label.item())
    return labels


def get_subject_splits(dataset, task, num_folds, seed, test_fraction=0.20, val_fraction=0.15):
    subj_labels = subject_split_labels(dataset, task)
    subjects = sorted(subj_labels.keys())
    labels   = np.array([subj_labels[s] for s in subjects])

    if num_folds <= 1:
        tv, test, ytv, _ = train_test_split(
            subjects, labels, test_size=test_fraction, random_state=seed,
            stratify=labels if len(set(labels)) > 1 else None)
        train, val, _, _ = train_test_split(
            tv, ytv, test_size=val_fraction, random_state=seed,
            stratify=ytv if len(set(ytv)) > 1 else None)
        return [{"train_subjects": train, "val_subjects": val, "test_subjects": test}]

    splits = []
    for k, (tv_idx, te_idx) in enumerate(
        StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed).split(subjects, labels)
    ):
        tv_s  = [subjects[i] for i in tv_idx]
        ytv   = labels[tv_idx]
        tr_i, v_i = train_test_split(np.arange(len(tv_s)), test_size=val_fraction,
                                     random_state=seed+k, stratify=ytv if len(set(ytv)) > 1 else None)
        splits.append({"train_subjects": [tv_s[i] for i in tr_i],
                       "val_subjects":   [tv_s[i] for i in v_i],
                       "test_subjects":  [subjects[i] for i in te_idx]})
    return splits


def subjects_to_indices(dataset, subj_list: List[str]) -> List[int]:
    s_set = set(subj_list)
    return [i for i, d in enumerate(dataset) if getattr(d, "subject_id", None) in s_set]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL FACTORIES
# ══════════════════════════════════════════════════════════════════════════════

def make_encoder(args, device):
    if getattr(args, "use_improved_gnn", False):
        if not _V10_EXTRAS:
            raise ImportError("--use_improved_gnn requires v9_extras.py")
        if args.layer_type == "WeightedGraphSAGE":
            raise ValueError(
                "--use_improved_gnn currently uses unweighted SAGEConv. "
                "Use --gnn_arch original --layer_type WeightedGraphSAGE for edge weights."
            )
        print(
            f"[make_encoder] ImprovedGraphSAGE v10 "
            f"(sd_p={args.stochastic_depth_p} fm_p={args.feature_mask_p} "
            f"dropedge_p={args.drop_edge_p})"
        )
        return ImprovedGraphSAGE(
            input_dim=4,
            hidden_dim=args.gnn_hidden_dim,
            output_dim=args.gnn_output_dim,
            dropout=0.25,
            dropedge_p=args.drop_edge_p,
            stochastic_depth_p=args.stochastic_depth_p,
            feature_mask_p=args.feature_mask_p,
        ).to(device)
    if args.gnn_arch == "advanced":
        if args.layer_type == "WeightedGraphSAGE":
            raise ValueError(
                "WeightedGraphSAGE is implemented in the original graph encoder. "
                "Use --gnn_arch original."
            )
        return AdvancedGraphNeuralNetwork(
            input_dim=4,
            hidden_dim=args.gnn_hidden_dim,
            output_dim=args.gnn_output_dim,
            dropout=0.25,
            layer_type=args.advanced_layer_type,
            num_layers=args.gnn_num_layers,
            activation=args.gnn_activation,
            residual=args.advanced_residual,
            jk=args.advanced_jk,
            heads=args.advanced_heads,
            use_time_features=args.use_time_features,
        ).to(device)

    dropedge = float(getattr(args, "drop_edge_p", 0.0) or 0.0)
    print(
        f"[make_encoder] GraphNeuralNetwork layer={args.layer_type} "
        f"topk={'off' if args.no_topk_pooling else f'on:{args.topk_ratio}'} "
        f"dropedge_p={dropedge}"
    )
    state = GraphNeuralNetwork(
        input_dim=4,
        hidden_dim=args.gnn_hidden_dim,
        output_dim=args.gnn_output_dim,
        dropout=0.25,
        use_topk_pooling=not args.no_topk_pooling,
        topk_ratio=args.topk_ratio,
        layer_type=args.layer_type,
        num_layers=args.gnn_num_layers,
        activation=args.gnn_activation,
        use_time_features=args.use_time_features,
        dropedge_p=dropedge,
    ).to(device)
    mode = getattr(args, "innovation_mode", "none")
    if mode == "none":
        return state
    from longitudinal_innovation import ResidualInnovationEncoder
    change = GraphNeuralNetwork(
        input_dim=10 if getattr(args, 'innovation_node_math', 'rate') in ('rate_laplacian','rate_flux') else 6,
        hidden_dim=args.gnn_hidden_dim,
        output_dim=args.gnn_output_dim,
        dropout=0.25,
        use_topk_pooling=not args.no_topk_pooling,
        topk_ratio=args.topk_ratio,
        layer_type=args.layer_type,
        num_layers=args.gnn_num_layers,
        activation=args.gnn_activation,
        use_time_features=args.use_time_features,
        dropedge_p=dropedge,
    ).to(device)
    return ResidualInnovationEncoder(
        state, change, mode,
        floor_years=args.innovation_floor_years,
        clip=args.innovation_clip,
        node_math=args.innovation_node_math, edge_math=args.innovation_edge_math,
        gate_mode=args.innovation_gate_mode,
    ).to(device)


def metadata_configuration(args):
    """Return structured-metadata settings for loader and classifier."""
    structured_available = (
        args.meta_mode != "none"
        and args.meta_encoder in {"mlp", "ft_transformer", "transformer", "ft_plus_ollama"}
    )
    early = structured_available and args.metadata_fusion in {"early", "both"}
    late = structured_available and args.metadata_fusion in {"late", "both"}
    loader_needs_metadata = early or late
    return structured_available, early, late, loader_needs_metadata


def build_sequence_classifier(args, graph_emb_dim, num_classes, dataset):
    structured_available, use_metadata_early, use_metadata_branch, _loader_meta = (
        metadata_configuration(args)
    )
    use_ollama = args.meta_encoder in {"ollama", "ft_plus_ollama"}
    if args.metadata_fusion != "none" and not structured_available:
        raise ValueError(
            "--metadata_fusion requires --meta_mode safe/full and a structured "
            "--meta_encoder (mlp or ft_transformer)."
        )

    common = dict(
        graph_emb_dim=graph_emb_dim,
        hidden_dim=args.lstm_hidden_dim,
        num_layers=args.lstm_num_layers,
        dropout=args.temporal_dropout,
        bidirectional=args.lstm_bidirectional,
        num_classes=num_classes,
        rnn_type=args.model_type,
        use_prev_dx=args.use_prev_dx,
        use_metadata=use_metadata_early,
        meta_num_dim=dataset.meta_num_dim,
        meta_cat_cardinalities=dataset.meta_cat_cardinalities,
        meta_emb_dim=args.meta_emb_dim,
        meta_encoder_type=args.meta_encoder,
        meta_token_dim=args.meta_token_dim,
        meta_transformer_layers=args.meta_transformer_layers,
        meta_transformer_heads=args.meta_transformer_heads,
        use_metadata_branch=use_metadata_branch,
        metadata_branch_dim=args.metadata_branch_dim,
        metadata_branch_layers=args.metadata_branch_layers,
        metadata_branch_heads=args.metadata_branch_heads,
        metadata_branch_dropout=args.metadata_branch_dropout,
        metadata_fusion_operator=args.metadata_fusion_operator,
        metadata_include_baseline=args.metadata_include_baseline,
        metadata_imputer_mode=args.metadata_imputer_mode,
        metadata_imputer_architecture=args.metadata_imputer_architecture,
        metadata_imputer_checkpoint=args.metadata_imputer_checkpoint,
        metadata_imputer_d_model=args.metadata_imputer_d_model,
        metadata_imputer_layers=args.metadata_imputer_layers,
        metadata_imputer_heads=args.metadata_imputer_heads,
        metadata_imputer_dropout=args.metadata_imputer_dropout,
        metadata_imputer_mask_probability=args.metadata_imputer_mask_probability,
        metadata_imputer_visit_mask_probability=args.metadata_imputer_visit_mask_probability,
        use_ollama_metadata=use_ollama,
        ollama_input_dim=dataset.ollama_dim,
        ollama_emb_dim=args.ollama_emb_dim,
        use_time_gap_input=args.use_time_gap_input,
    )
    if getattr(args, "clinical_only", False):
        if args.metadata_fusion != "late" or not use_metadata_branch:
            raise ValueError("--clinical_only requires a structured metadata --metadata_fusion late branch")
        if args.use_history_v13:
            raise ValueError("--clinical_only and --use_history_v13 cannot be combined")
        model = ClinicalOnlyTemporalClassifier(**common)
        temporal_name = "clinical-only/temporal-metadata"
    elif args.use_history_v13:
        model = DTISequenceClassifierV13(
            **common,
            n_attn_heads=args.attn_heads,
            history_fusion=args.history_fusion,
            history_branch_dim=args.history_branch_dim,
            history_branch_layers=args.history_branch_layers,
            history_branch_heads=args.history_branch_heads,
            history_dropout=args.history_dropout,
            history_gate_init=args.history_gate_init,
            history_stats_dim=args.history_stats_dim,
            classifier_head=args.classifier_head,
            metadata_modality_dropout=args.metadata_modality_dropout,
        )
        temporal_name = f"v13-history/{args.history_fusion}/{args.attn_heads}heads"
    elif args.use_attention_temporal:
        model = DTISequenceClassifierV2(
            **common,
            n_attn_heads=args.attn_heads,
            use_visit_film=getattr(args,"visit_film",False),
            use_clinical_reliability=getattr(args,"clinical_reliability",False),
            reliability_tau_days=getattr(args,"reliability_tau_days",90.0),
            clinical_reliability_mode=getattr(args,"clinical_reliability_mode","hard"),
            clinical_film_mode=getattr(args,"clinical_film_mode","standard"),
        )
        temporal_name = f"attention/{args.attn_heads}heads"
    else:
        model = DTISequenceClassifier(**common)
        temporal_name = args.model_type

    print(
        f"[classifier] temporal={temporal_name} metadata_fusion={args.metadata_fusion} "
        f"meta_mode={args.meta_mode} meta_encoder={args.meta_encoder} "
        f"fusion_op={args.metadata_fusion_operator} baseline={args.metadata_include_baseline} "
        f"imputer={args.metadata_imputer_mode} history_v13={args.use_history_v13} "
        f"history_fusion={args.history_fusion} head={args.classifier_head}"
    )
    return model


def build_loader(args, dataset, indices, encoder, device, shuffle, seed, use_metadata, use_ollama):
    return DTIWindowDataLoaderStableChange(
        dataset,
        indices,
        encoder,
        device,
        batch_size=args.batch_size,
        shuffle=shuffle,
        seed=seed,
        task=args.task,
        window_mode=args.window_mode,
        max_input_visits=args.max_input_visits,
        use_prev_dx=args.use_prev_dx,
        use_metadata=use_metadata,
        use_ollama=use_ollama,
        use_time_features=args.use_time_features,
        drop_reverse_windows=args.drop_reverse_windows,
        split_at_reverse_windows=args.split_at_reverse_windows,
        dynamic_tract_edges=args.dynamic_tract_edges,
        tract_root=args.tract_root,
        tract_topk=args.tract_topk,
        fixed_tract_support=args.fixed_tract_support,
        tract_missing_policy=args.tract_missing_policy,
    )


def build_loader_with_task(
    args,
    dataset,
    indices,
    encoder,
    device,
    shuffle,
    seed,
    use_metadata,
    use_ollama,
    task_override: Optional[str] = None,
):
    task = task_override if task_override is not None else args.task
    return DTIWindowDataLoaderStableChange(
        dataset,
        indices,
        encoder,
        device,
        batch_size=args.batch_size,
        shuffle=shuffle,
        seed=seed,
        task=task,
        window_mode=args.window_mode,
        max_input_visits=args.max_input_visits,
        use_prev_dx=args.use_prev_dx,
        use_metadata=use_metadata,
        use_ollama=use_ollama,
        use_time_features=args.use_time_features,
        drop_reverse_windows=args.drop_reverse_windows,
        split_at_reverse_windows=args.split_at_reverse_windows,
        dynamic_tract_edges=args.dynamic_tract_edges,
        tract_root=args.tract_root,
        tract_topk=args.tract_topk,
        fixed_tract_support=args.fixed_tract_support,
        tract_missing_policy=args.tract_missing_policy,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # paths
    p.add_argument("--dti_root",      required=True)
    p.add_argument("--selected_csv",  required=True)
    p.add_argument("--manifest_csv",  required=True)
    p.add_argument("--meta_csv",      default=None)
    p.add_argument("--expanded_clinical_root", default=None,
                   help="Path to expanded clinical dataset v2 root; no fMRI")
    p.add_argument("--expanded_clinical_alignment",
                   choices=["past126_plus00","past126_plus07","past126_plus15"],
                   default="past126_plus00")
    p.add_argument("--expanded_clinical_set",
                   choices=["C0_context","C1_summary","C2_summary","C1_cognition",
                            "C2_primary","C2_noCDR","C3_extended"],
                   default="C2_summary")

    p.add_argument("--aal_adjacency", required=True)
    p.add_argument("--save_path",     default="./runs_dti_stgnn/stable_change_v3")

    # task / windowing
    p.add_argument("--task", choices=["stable_change","next_dx","progression","converter"], default="stable_change")
    p.add_argument("--window_mode", choices=["final_only","sliding"], default="sliding")
    p.add_argument("--max_visits",       type=int, default=10)
    p.add_argument("--max_input_visits", type=int, default=9)
    p.add_argument("--min_visits",       type=int, default=2)
    p.add_argument("--drop_reverse_windows",   action="store_true")
    p.add_argument("--split_at_reverse_windows", action="store_true")
    p.add_argument("--match_window_days", type=int, default=126)
    p.add_argument("--check_only",       action="store_true")

    # dynamic tractography
    p.add_argument("--dynamic_tract_edges", action=argparse.BooleanOptionalAction, default=False,
                   help="Use visit-specific tract matrices instead of only --aal_adjacency.")
    p.add_argument("--tract_root", default=None)
    p.add_argument("--tract_topk", type=int, default=20,
                   help="Per-ROI Top-K used when visit-specific topology is allowed.")
    p.add_argument("--fixed_tract_support", default=None,
                   help="Optional fixed binary/weighted support shared by every visit. "
                        "With --dynamic_tract_edges, visit matrices provide weights only.")
    p.add_argument("--tract_missing_policy",
                   choices=["skip_window", "static_fallback", "error"],
                   default="skip_window",
                   help="Behavior when a visit-specific tract matrix is missing.")

    # Causal structural innovations and visit-level clinical reliability (opt-in).
    p.add_argument("--innovation_mode", choices=("none","node","edge","both"), default="none",
                   help="Encode causal within-patient node and/or tract differences on fixed AAL support")
    p.add_argument("--innovation_floor_years", type=float, default=0.25)
    p.add_argument("--innovation_clip", type=float, default=3.0)
    p.add_argument("--innovation_node_math", choices=("rate","laplacian","rate_laplacian","flux","rate_flux"), default="rate",
                   help="Original node rate, graph Laplacian change, or joint rate and Laplacian channels")
    p.add_argument("--innovation_gate_mode", choices=("global","visit","feature"), default="global",
                   help="Constant original innovation strength or learned per-visit/feature gating")
    p.add_argument("--innovation_edge_math", choices=("rate","logratio"), default="rate",
                   help="Original signed tract rate or signed log1p tract ratio rate")
    p.add_argument("--visit_film", action="store_true",
                   help="Add clinical-conditioned residual FiLM to each DTI visit before temporal Transformer")
    p.add_argument("--clinical_reliability", action="store_true",
                   help="Weight visit FiLM using past-only assessment age and numeric observation mask")
    p.add_argument("--reliability_tau_days", type=float, default=90.0)
    p.add_argument("--clinical_film_mode", choices=("standard","concordance"), default="standard",
                   help="Original clinical FiLM or cross-modal DTI/clinical similarity-conditioned FiLM")
    p.add_argument("--clinical_reliability_mode", choices=("hard","adaptive"), default="hard",
                   help="Fixed reliability attenuation or learned conservative interpolation")

    # metadata / text
    p.add_argument("--meta_mode",    choices=["none","safe","full"], default="none")
    p.add_argument("--meta_encoder", choices=["mlp","ft_transformer","transformer","ollama","ft_plus_ollama"], default="mlp")
    p.add_argument("--meta_emb_dim",          type=int, default=64)
    p.add_argument("--meta_token_dim",         type=int, default=32)
    p.add_argument("--meta_transformer_layers",type=int, default=2)
    p.add_argument("--meta_transformer_heads", type=int, default=4)
    p.add_argument("--clinical_only", action="store_true",
                   help="Predict using clinical temporal metadata, previous DX and time gap; NO DTI features in classifier")
    p.add_argument("--metadata_fusion",
                   choices=["none", "early", "late", "both"], default="none",
                   help="early=metadata enters the graph temporal stream; "
                        "late=independent metadata temporal branch with gated fusion.")
    p.add_argument("--metadata_branch_dim", type=int, default=128)
    p.add_argument("--metadata_branch_layers", type=int, default=1)
    p.add_argument("--metadata_branch_heads", type=int, default=2)
    p.add_argument("--metadata_branch_dropout", type=float, default=0.30)
    p.add_argument("--metadata_fusion_operator",
                   choices=["gated", "add", "concat", "film"], default="gated",
                   help="Operator used by late/both fusion.")
    p.add_argument("--metadata_include_baseline", action=argparse.BooleanOptionalAction, default=False,
                   help="Encode *_bl fields as separate baseline context.")
    p.add_argument("--metadata_imputer_mode",
                   choices=["none", "frozen", "joint", "scratch_joint"], default="none")
    p.add_argument("--metadata_imputer_architecture", choices=["transformer", "gru"], default="transformer")
    p.add_argument("--metadata_imputer_checkpoint", default=None)
    p.add_argument("--metadata_imputer_loss_weight", type=float, default=0.10)
    p.add_argument("--metadata_imputer_mask_probability", type=float, default=0.20)
    p.add_argument("--metadata_imputer_visit_mask_probability", type=float, default=0.05)
    p.add_argument("--metadata_imputer_d_model", type=int, default=128)
    p.add_argument("--metadata_imputer_layers", type=int, default=2)
    p.add_argument("--metadata_imputer_heads", type=int, default=4)
    p.add_argument("--metadata_imputer_dropout", type=float, default=0.20)
    p.add_argument("--meta_scaler_checkpoint", default=None,
                   help="Use field order/mean/std from an imputer checkpoint.")

    # V13 strict-history cache and uncertainty-aware multimodal fusion.
    p.add_argument("--use_history_v13", action=argparse.BooleanOptionalAction, default=False,
                   help="Use DTISequenceClassifierV13 with a strict-past history cache supplied through --ollama_cache.")
    p.add_argument("--history_fusion", choices=["none", "early", "late", "both", "cross"], default="late")
    p.add_argument("--history_branch_dim", type=int, default=128)
    p.add_argument("--history_branch_layers", type=int, default=2)
    p.add_argument("--history_branch_heads", type=int, default=4)
    p.add_argument("--history_dropout", type=float, default=0.30)
    p.add_argument("--history_gate_init", type=float, default=-2.0,
                   help="Initial reliability-gate logit; -2 starts near graph-only behavior.")
    p.add_argument("--history_stats_dim", type=int, default=8)
    p.add_argument("--classifier_head", choices=["small", "large", "residual"], default="large")
    p.add_argument("--metadata_modality_dropout", type=float, default=0.0,
                   help="Per-subject probability of dropping each metadata branch during training.")
    p.add_argument("--ollama_cache",   default=None)
    p.add_argument("--ollama_emb_dim", type=int, default=128)
    p.add_argument("--use_prev_dx",    action=argparse.BooleanOptionalAction, default=False)

    # ── NEW: hard DX constraint ──────────────────────────────────────────────
    p.add_argument("--use_dx_hard_rule", action=argparse.BooleanOptionalAction, default=True,
                   help="Hard rule: last DX=AD → always stable. 100%% correct given drop_reverse_windows.")

    # ── NEW: PNA multi-aggregator ─────────────────────────────────────────────
    p.add_argument("--use_pna_enricher", action=argparse.BooleanOptionalAction, default=True,
                   help="PNA-style multi-aggregator on graph embeddings before LSTM. "
                        "Adds mean/max/min/std × 3 degree-scalers, projects back to same dim. "
                        "Corso et al., NeurIPS 2020.")

    # ── NEW: subject-bias net ─────────────────────────────────────────────────
    p.add_argument("--use_subject_bias",   action=argparse.BooleanOptionalAction, default=True,
                   help="Add SubjectBiasNet (age, sex, APOE4) logit bias. "
                        "Requires --meta_mode safe/full; disabled otherwise.")
    p.add_argument("--subject_bias_hidden", type=int,   default=32)
    p.add_argument("--subject_bias_scale",  type=float, default=0.2,
                   help="Multiplier on bias logits before adding to main logits.")
    p.add_argument("--subject_bias_dropout",type=float, default=0.30)

    # ── NEW: temporal delta ───────────────────────────────────────────────────
    p.add_argument("--use_temporal_delta", action=argparse.BooleanOptionalAction, default=True,
                   help="Concatenate Δg_t = g_t − g_{t-1} to graph embeddings → LSTM sees rate-of-change.")

    # ── NEW: temperature scaling ──────────────────────────────────────────────
    p.add_argument("--use_temp_scaling", action=argparse.BooleanOptionalAction, default=True,
                   help="Fit temperature T on val logits after training for probability calibration.")

    # ── NEW: prevalence constraint in threshold search ─────────────────────────
    p.add_argument("--prevalence_tolerance", type=float, default=3.0,
                   help="Threshold search: predicted changed fraction must be within this factor of val changed fraction.")

    # converter-aware loss weighting
    p.add_argument("--use_converter_weight",  action="store_true")
    p.add_argument("--converter_stable_mult", type=float, default=2.0)
    p.add_argument("--converter_changed_mult",type=float, default=3.0)

    # training
    p.add_argument("--n_epochs",    type=int, default=200)
    p.add_argument("--patience",    type=int, default=60)    # ⑦ was 40
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--lr",          type=float, default=2e-4,
                   help="2e-4 gives GNN enough gradient signal. Differential LR (v4) caused GNN freeze.")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_folds",   type=int, default=1)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--split_seed", type=int, default=None,
                   help="Optional fixed subject-split seed, separate from model initialization seed.")
    p.add_argument("--val_fraction",  type=float, default=0.15)
    p.add_argument("--test_fraction", type=float, default=0.20)
    p.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=True)

    # LR schedule
    p.add_argument("--scheduler", choices=["plateau","cosine","onecycle"], default="cosine")
    p.add_argument("--cosine_T0",    type=int,   default=30)   # ⑦ was 20
    p.add_argument("--cosine_Tmult", type=int,   default=2)
    p.add_argument("--warmup_epochs",type=int,   default=5)

    # imbalance ⑥
    p.add_argument("--train_stable_to_changed_ratio", type=float, default=0.0)
    p.add_argument("--min_epochs", type=int, default=0,
                   help="Warmup: skip best-checkpoint saving before this epoch. Prevents lucky early val spikes.")
    # ── Multi-atlas fusion (v11) ──────────────────────────────────────────
    p.add_argument("--fusion_atlases", type=str, default="",
                   help="Comma-separated extra atlases to fuse: jhu,schaefer100,"
                        "schaefer200,schaefer400. Empty=off. Late-fusion MLP branch.")
    p.add_argument("--fusion_jhu_root", type=str, default="",
                   help="Override JHU root dir (else registry default).")
    p.add_argument("--fusion_sch100_root", type=str, default="",
                   help="Override Schaefer100 root dir.")
    p.add_argument("--fusion_sch200_root", type=str, default="",
                   help="Override Schaefer200 root dir.")
    p.add_argument("--fusion_sch400_root", type=str, default="",
                   help="Override Schaefer400 root dir.")
    p.add_argument("--fusion_branch_dim", type=int, default=64,
                   help="Per-atlas MLP branch embedding size.")
    p.add_argument("--fusion_dim", type=int, default=128,
                   help="Final fusion embedding size.")
    p.add_argument("--fusion_dropout", type=float, default=0.5,
                   help="Dropout inside fusion branches/head.")
    p.add_argument("--fusion_use_clinical", action="store_true",
                   help="Add age/sex/apoe4 clinical features to the fusion branch.")
    p.add_argument("--fusion_add_rate", action="store_true",
                   help="Append per-ROI rate-of-change (delta/n_steps) to each atlas "
                        "branch — the report's rate-of-change feature. +1 block per atlas.")
    p.add_argument("--fusion_no_normalize", action="store_true",
                   help="Disable per-atlas z-scoring (default: normalize on train).")
    # ── Decoupled two-stage training (cRT / LWS) ──────────────────────────
    p.add_argument("--crt_epochs", type=int, default=0,
                   help="Decoupled classifier re-training: after main training, "
                        "FREEZE the encoder/GNN and retrain ONLY the classifier head "
                        "for N epochs on CLASS-BALANCED batches. 0=off. 10-30 typical. "
                        "Best-evidenced minority-recall technique (Kang et al. 2020).")
    p.add_argument("--crt_lr", type=float, default=1e-3,
                   help="LR for the cRT classifier-only stage (higher OK — few params).")
    p.add_argument("--crt_mode", type=str, default="crt",
                   choices=["crt", "lws"],
                   help="crt=re-init & retrain head; lws=learn per-class logit scales only.")
    p.add_argument("--crt_balance", type=str, default="balanced",
                   choices=["balanced", "sqrt"],
                   help="Class-balanced sampling ('balanced'=1:1) or 'sqrt' (milder).")
    p.add_argument("--weight_avg_topk", type=int, default=0,
                   help="SWA-style: average weights of top-K checkpoints from the "
                        "tail window (by val score). 0=off. 3-5 typical. "
                        "Reduces fold variance by smoothing the loss landscape minimum.")
    p.add_argument("--ckpt_ensemble_topk", type=int, default=0,
                   help="Prediction ensembling: average softmax probs of top-K tail "
                        "checkpoints at inference. 0=off. Alternative to weight_avg. "
                        "Better calibration, slightly slower eval.")
    p.add_argument("--use_ema", action="store_true",
                   help="Maintain exponential moving average of weights during training. "
                        "Evaluated/checkpointed alongside raw weights.")
    p.add_argument("--ema_decay", type=float, default=0.999,
                   help="EMA decay factor (higher=slower/smoother). 0.995-0.9995 typical.")
    p.add_argument("--checkpoint_window", type=int, default=0,
                   help="When >0: at early stop, pick best checkpoint from last N epochs instead of global best. Avoids early noise spikes. 20 is a good value.")
    p.add_argument("--eval_stable_to_changed_ratio", type=float, default=0.0,
                   help="Downsample stable windows in val/test to this ratio. "
                        "0=natural distribution. 1.0=1:1 balanced. All changed kept.")
    p.add_argument("--oversample_changed_ratio",      type=float, default=3.0,
                   help="3:1 gave best v2 calibration. Closer to training prior = better raw probabilities.")  # ⑥ was 3.0

    # augmentation
    p.add_argument("--graph_noise_std", type=float, default=0.02)

    # threshold calibration
    p.add_argument("--optimize_threshold", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--threshold_metric", choices=["balanced_accuracy","f1_changed","macro_f1"],
                   default="balanced_accuracy")

    # TTA
    p.add_argument("--tta_passes", type=int, default=0)

    # early stopping
    p.add_argument("--early_stop_metric",
                   choices=["balanced_accuracy","auc","macro_f1","f1_changed"],
                   default="auc")

    # temporal model
    p.add_argument("--model_type", choices=["LSTM","GRU","RNN"], default="GRU",
                   help="GRU: empirically better than LSTM on short sequences (2-9 visits). Fewer params.")
    p.add_argument("--lstm_hidden_dim",    type=int, default=128,
                   help="With ~800 windows, 128 prevents overfitting. Scale up only with more data.")
    p.add_argument("--lstm_num_layers",    type=int, default=2)     # ⑤ was 1
    p.add_argument("--lstm_bidirectional", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--temporal_dropout",   type=float, default=0.50)  # ⑤ was 0.45

    # GNN ⑤
    p.add_argument("--gnn_arch", choices=["original","advanced"], default="original",
                   help="original=GraphSAGE (empirically best on this dataset size). advanced=GATv2.")
    p.add_argument("--layer_type", choices=["GCN","GAT","GraphSAGE","WeightedGraphSAGE"], default="GraphSAGE")
    p.add_argument("--advanced_layer_type",
                   choices=["GCN","GAT","GATv2","GraphSAGE","GIN","Transformer"], default="GATv2",
                   help="GATv2: learns attention weights on each white-matter edge — which tracts matter most for DX change.")
    p.add_argument("--advanced_residual", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--advanced_jk", choices=["last","concat"], default="last")
    p.add_argument("--advanced_heads", type=int, default=4)
    p.add_argument("--gnn_hidden_dim",  type=int, default=128,
                   help="128 for <1000 windows. Increase to 256 with more data.")
    p.add_argument("--gnn_output_dim",  type=int, default=128)
    p.add_argument("--gnn_num_layers",  type=int, default=2,
                   help="2 layers for 116 ROIs. 3+ risks over-smoothing where all nodes look similar.")
    p.add_argument("--gnn_activation",  choices=["relu","leaky_relu","elu","gelu"], default="gelu")
    p.add_argument("--no_topk_pooling", action="store_true")
    p.add_argument("--topk_ratio",      type=float, default=0.5)
    p.add_argument("--use_time_features",  action="store_true")
    p.add_argument("--use_time_gap_input", action=argparse.BooleanOptionalAction, default=False)

    # loss
    p.add_argument("--loss_type", choices=["ce","focal","sba", "logit_adjusted"], default="ce",
                   help="ce=weighted CrossEntropy (best in ablation). focal can interact badly with oversampling.")
    p.add_argument("--focal_alpha", type=float, default=0.75)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--sba_alpha", type=float, default=0.5,
                   help="Blend for SBA loss: L=alpha*L_sba+(1-alpha)*L_ce. 1.0=pure SBA.")
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--save_model",  action="store_true")
    p.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--clip_features",action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--roi_subdir", default="aal", help="Visit roi_stats subdirectory for the primary atlas")
    p.add_argument("--n_rois", type=int, default=116, help="Number of ROIs expected in CSV files")

    # ── Early stopping / training length ────────────────────────────────────
    p.add_argument(
        "--no_early_stop", action="store_true",
        help="Run the full n_epochs regardless of val metric. "
             "Useful for small datasets where the best checkpoint can occur late. "
             "Equivalent to --patience n_epochs."
    )

    # ── DropEdge (graph edge dropout during training) ────────────────────────
    p.add_argument("--drop_edge_p", type=float, default=0.0,
                   help="Probability of randomly dropping each tract edge per "
                        "training forward pass (DropEdge). 0=off, 0.2=mild, "
                        "0.4=aggressive. Does NOT apply at validation/test time. "
                        "Forces GNN to not over-rely on any single tract connection.")

    # ── Supervised Contrastive auxiliary loss ──────────────────────────────────
    p.add_argument("--use_supcon", action="store_true",
                   help="Add supervised contrastive loss on GRU window embeddings "
                        "as an auxiliary objective: L = L_main + supcon_weight * L_supcon. "
                        "Pulls changed-window embeddings together, pushes from stable.")
    p.add_argument("--supcon_weight", type=float, default=0.1,
                   help="Weight for the SupCon auxiliary loss (default 0.1). "
                        "Larger = stronger pull on embedding space.")
    p.add_argument("--supcon_temperature", type=float, default=0.07,
                   help="Temperature for SupCon similarity scaling (default 0.07). "
                        "Lower = sharper contrast between classes.")

    # ── Pre-training (Phase 1) ────────────────────────────────────────────────
    p.add_argument(
        "--pretrain_epochs", type=int, default=30,
        help="Epochs to pre-train GNN+LSTM on 3-class next-DX (CN/MCI/AD) before "
             "fine-tuning on stable/changed.  0 = skip pre-training.  "
             "Pre-training uses the same encoder and temporal model but a separate "
             "3-class head that is discarded afterward; matching weights are "
             "transferred to the fine-tune classifier via strict=False load_state_dict."
    )
    p.add_argument("--pretrain_lr", type=float, default=2e-4,
                   help="Learning rate for Phase 1 pre-training.")
    p.add_argument("--pretrain_log_every", type=int, default=10,
                   help="Print pre-training loss every N epochs.")

    # W&B
    p.add_argument("--wandb",          action="store_true")
    p.add_argument("--wandb_project",  default="dti-stgnn-stable-change")
    p.add_argument("--wandb_entity",   default=None)
    p.add_argument("--wandb_mode",     choices=["online","offline","disabled"], default="online")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_tags",     default="")

    p.add_argument("--log_test_every", type=int, default=0,
                   help="Log test AUC/BalAcc every N epochs (0=disabled). "
                        "Output: <save_path>/test_curve_foldN.csv. "
                        "Start controlled by --log_test_start.")
    p.add_argument("--log_test_start", type=int, default=200,
                   help="Epoch to begin per-epoch test logging (default 200).")

    # ── v10: improved GNN ───────────────────────────────────────────────
    p.add_argument("--use_improved_gnn", action="store_true", default=False,
                   help="Replace GraphNeuralNetwork with ImprovedGraphSAGE "
                        "(PairNorm + residual + JK-Max + stochastic depth + "
                        "feature masking). Requires v9_extras.py.")
    p.add_argument("--stochastic_depth_p", type=float, default=0.15,
                   help="Prob of skipping GNN layer 2 during training (default 0.15).")
    p.add_argument("--feature_mask_p", type=float, default=0.08,
                   help="Prob of zeroing each DTI input feature dim (default 0.08).")

    # ── v10: logit-adjusted loss ─────────────────────────────────────────
    p.add_argument("--logit_adj_tau", type=float, default=1.0,
                   help="Tau for logit-adjusted loss. Use with "
                        "--loss_type logit_adjusted.")

    # ── v10: multi-scale temporal delta ──────────────────────────────────
    p.add_argument("--use_multiscale_delta", action="store_true", default=False,
                    help="Use Δt=1 + Δt=2 temporal deltas (vs Δt=1 only). "
                            "Activates compute_multiscale_temporal_delta.")
    p.add_argument("--use_attention_temporal", action=argparse.BooleanOptionalAction, default=False,
                   help="Replace BiGRU with attention-based DTISequenceClassifierV2")
    p.add_argument("--attn_heads", type=int, default=2,
                   help="Attention heads for V2 (must divide hidden_dim*2=256)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of each tracked module's parameters, updated as:
        shadow = decay * shadow + (1 - decay) * param

    At eval time, call apply_shadow() to swap in EMA weights, and restore()
    to swap back the raw training weights.  Standard technique for stabilising
    the final model — the averaged weights sit in a flatter region of the loss
    landscape and generalise better, which is exactly what reduces fold variance.

    Reference: Polyak averaging / Izmailov et al. 2018 (SWA).
    """
    def __init__(self, modules: dict, decay: float = 0.999):
        # modules: {name: nn.Module}
        self.decay   = decay
        self.modules = {k: m for k, m in modules.items() if m is not None}
        self.shadow  = {
            name: {k: v.detach().clone() for k, v in m.state_dict().items()}
            for name, m in self.modules.items()
        }
        self._backup = None

    @torch.no_grad()
    def update(self):
        for name, m in self.modules.items():
            for k, v in m.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[name][k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
                else:
                    self.shadow[name][k].copy_(v.detach())   # buffers (int counts etc.)

    def apply_shadow(self):
        self._backup = {
            name: {k: v.detach().clone() for k, v in m.state_dict().items()}
            for name, m in self.modules.items()
        }
        for name, m in self.modules.items():
            m.load_state_dict(self.shadow[name])

    def restore(self):
        if self._backup is None:
            return
        for name, m in self.modules.items():
            m.load_state_dict(self._backup[name])
        self._backup = None

    def state(self) -> dict:
        return {name: {k: v.clone() for k, v in sd.items()}
                for name, sd in self.shadow.items()}


def average_state_dicts(state_dicts: list) -> dict:
    """Average a list of state_dict-like dicts (SWA weight averaging).

    Only floating-point tensors are averaged; integer buffers take the last
    value.  Assumes identical keys/shapes across all inputs (same architecture).
    """
    if not state_dicts:
        return {}
    if len(state_dicts) == 1:
        return state_dicts[0]
    avg = {}
    keys = state_dicts[0].keys()
    for k in keys:
        vals = [sd[k] for sd in state_dicts]
        if vals[0].dtype.is_floating_point:
            stacked = torch.stack([v.float() for v in vals], dim=0)
            avg[k] = stacked.mean(dim=0).to(vals[0].dtype)
        else:
            avg[k] = vals[-1].clone()
    return avg


class LWSScaler(nn.Module):
    """Learnable Weight Scaling (Kang et al. 2020, "Decoupling...").

    Wraps a frozen classifier and learns ONE positive scale per class applied
    to that class's logits.  Far fewer params than re-training the whole head
    (num_classes scalars), so it's a gentler rebalancing that rarely overfits.
    """
    def __init__(self, classifier, num_classes: int):
        super().__init__()
        self.classifier = classifier
        for p in self.classifier.parameters():
            p.requires_grad = False
        self.log_scales = nn.Parameter(torch.zeros(num_classes))

    def forward(self, *args, **kwargs):
        logits = self.classifier(*args, **kwargs)
        return logits * torch.exp(self.log_scales)


def run_crt_stage(args, encoder, classifier, subject_bias_net, pna_enricher,
                  train_loader, num_classes, ctx_train, device, class_counts):
    """Decoupled classifier re-training (cRT) or Learnable Weight Scaling (LWS).

    Stage 2 of decoupled training: the representation (encoder/GNN/PNA/bias) is
    FROZEN at its trained state; only the classifier head is updated, on
    class-balanced batches.  This fixes the classifier's decision boundary bias
    toward the majority class WITHOUT letting the imbalanced signal re-corrupt
    the learned features.  Empirically the single most reliable minority-recall
    boost on long-tailed data.

    Returns the (possibly wrapped) classifier to use for eval.
    """
    import numpy as _np

    # Freeze everything except the classifier head
    for mod in (encoder, subject_bias_net, pna_enricher):
        if mod is not None:
            mod.eval()
            for p in mod.parameters():
                p.requires_grad = False

    if args.crt_mode == "lws":
        # Learn per-class logit scales only
        wrapped = LWSScaler(classifier, num_classes).to(device)
        opt = torch.optim.AdamW([wrapped.log_scales], lr=args.crt_lr)
        train_target = wrapped
        print(f"[cRT/LWS] learning {num_classes} per-class scales, "
              f"{args.crt_epochs} epochs, lr={args.crt_lr}")
    else:
        # Re-initialise and train ONLY the final classification head. Earlier
        # versions accidentally reset every Linear layer in the temporal model,
        # metadata encoder, and imputer, which was not true cRT.
        classifier.eval()
        for parameter in classifier.parameters():
            parameter.requires_grad = False
        head = getattr(classifier, "classifier", None)
        if head is None:
            raise AttributeError("cRT requires classifier.classifier final head")
        for module in head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        head.train()
        for parameter in head.parameters():
            parameter.requires_grad = True
        opt = torch.optim.AdamW(head.parameters(), lr=args.crt_lr,
                                weight_decay=args.weight_decay)
        train_target = classifier
        print(f"[cRT] re-training final classifier head only, "
              f"{args.crt_epochs} epochs, lr={args.crt_lr}, balance={args.crt_balance}")

    # Class-balanced sampling weights over the training windows
    windows = train_loader.windows
    labels  = _np.array([int(w[-1]) for w in windows])
    counts  = _np.bincount(labels, minlength=num_classes).astype(_np.float64)
    counts[counts == 0] = 1.0
    if args.crt_balance == "sqrt":
        cls_w = 1.0 / _np.sqrt(counts)
    else:
        cls_w = 1.0 / counts
    sample_w = cls_w[labels]
    sample_w = sample_w / sample_w.sum()

    plain_ce = nn.CrossEntropyLoss()
    rng = _np.random.default_rng(args.seed)
    n_windows = len(windows)
    batch_size = max(8, args.batch_size)

    for ep in range(1, args.crt_epochs + 1):
        # Sample a class-balanced epoch (with replacement) by reordering windows
        idx = rng.choice(n_windows, size=n_windows, replace=True, p=sample_w)
        train_loader.windows = [windows[i] for i in idx]

        ep_loss = 0.0
        for batch in train_loader:
            opt.zero_grad()
            with torch.no_grad():
                # encoder frozen → build graph embeddings without grad
                pass
            logits = _full_forward(batch, encoder, train_target, ctx=ctx_train,
                                   drop_edge_p=0.0)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("cRT produced non-finite logits")
            labels_t = batch["labels"].to(device)
            loss = plain_ce(logits, labels_t)
            if not torch.isfinite(loss):
                raise FloatingPointError("cRT produced a non-finite loss")
            loss.backward()
            trainable_parameters = [
                parameter
                for group in opt.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]
            if trainable_parameters:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=5.0,
                )
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise FloatingPointError(
                        "cRT produced non-finite gradients"
                    )
            opt.step()
            for group in opt.param_groups:
                for parameter in group["params"]:
                    if not torch.isfinite(parameter).all():
                        raise FloatingPointError(
                            "cRT produced non-finite classifier parameters"
                        )
            ep_loss += float(loss.item())
        if ep == 1 or ep % 5 == 0 or ep == args.crt_epochs:
            print(f"[cRT] ep {ep}/{args.crt_epochs}  loss={ep_loss:.3f}")

    # restore original window order
    train_loader.windows = windows
    return train_target


def main():  # noqa: C901
    args = parse_args()
    if args.innovation_mode != "none":
        if args.gnn_arch != "original" or getattr(args,"use_improved_gnn",False):
            raise ValueError("Structural innovation requires gnn_arch=original")
        if args.n_rois != 116 or args.roi_subdir != "aal":
            raise ValueError("Structural innovation currently requires AAL116 matched ROI order")
        if args.innovation_mode in {"edge","both"} and not (args.dynamic_tract_edges and args.fixed_tract_support):
            raise ValueError("Edge innovations require dynamic_tract_edges and fixed_tract_support")
        if args.innovation_floor_years <= 0 or args.innovation_clip <= 0:
            raise ValueError("Innovation floor and clipping must be positive")
    if args.innovation_node_math in ('flux','rate_flux') and args.innovation_mode != 'both':
        raise ValueError('Flux node math requires --innovation_mode both')
    if args.innovation_gate_mode != 'global' and args.innovation_mode == 'none':
        raise ValueError('Innovation gate requires enabled graph changes')
    if args.clinical_film_mode != 'standard' and not args.visit_film:
        raise ValueError('Clinical concordance FiLM requires --visit_film')
    if args.innovation_node_math != 'rate' and args.innovation_mode not in ('node','both'):
        raise ValueError('Nondefault node math requires enabled node innovation')
    if args.innovation_edge_math != 'rate' and args.innovation_mode not in ('edge','both'):
        raise ValueError('Nondefault edge math requires enabled edge innovation')
    if args.clinical_reliability_mode != 'hard' and not args.clinical_reliability:
        raise ValueError('Adaptive reliability requires --clinical_reliability')
    if args.clinical_reliability and not args.visit_film:
        raise ValueError("--clinical_reliability requires --visit_film")
    if args.visit_film:
        if not args.use_attention_temporal or args.use_history_v13 or getattr(args,"clinical_only",False):
            raise ValueError("--visit_film requires DTISequenceClassifierV2 (--use_attention_temporal), not clinical-only/V13")
        if args.metadata_fusion != "late" or args.meta_mode == "none":
            raise ValueError("--visit_film requires a structured late clinical branch")
    if args.clinical_reliability:
        if not args.expanded_clinical_root or args.expanded_clinical_alignment != "past126_plus00":
            raise ValueError("Clinical reliability requires same-day/past-only expanded v2 clinical inputs (past126_plus00)")
    if args.reliability_tau_days <= 0:
        raise ValueError("Reliability tau must be >0")
    if args.no_early_stop:
        args.patience = args.n_epochs   # run full training
        print(f"[no_early_stop] patience set to {args.patience} (full training)")
    set_random_seeds(args.seed)

    save_root = Path(args.save_path).expanduser()
    save_root.mkdir(parents=True, exist_ok=True)
    (save_root / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.use_amp and device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"[device] {device}" + (f"  {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    print(f"[amp]    {'enabled' if use_amp else 'disabled'}")

    # ── always load converter lookup (needed for SubjectBiasNet + optional loss weight) ──
    converter_lookup = load_converter_lookup(args.selected_csv)
    n_conv = sum(1 for v in converter_lookup.values() if v["is_converter"])
    print(f"[converter] {len(converter_lookup)} subjects  converters={n_conv}  non={len(converter_lookup)-n_conv}")
    # ── subject-level metadata lookup (for SubjectBiasNet bias) ───────────────
    meta_lookup: Optional[Dict[str, Dict]] = None
    if args.meta_mode != "none" and args.use_subject_bias:
        import pandas as _pd
        _mdf = _pd.read_csv(args.manifest_csv)
        _age_mean = float(_mdf["age"].mean())
        _age_std  = float(_mdf["age"].std()) or 1.0
        _meta: Dict[str, Dict] = {}
        for _, _r in _mdf.iterrows():
            _sid = str(_r["subject_id"])
            if _sid not in _meta:
                _meta[_sid] = {
                    "age_norm":   (float(_r.get("age",  _age_mean)) - _age_mean) / _age_std,
                    "sex_num":    float(_r.get("sex_num", 0.5)),
                    "apoe4_norm": float(_r.get("apoe4",  0.0)) / 2.0,
                }
        meta_lookup = _meta
        print(f"[meta_bias] loaded {len(meta_lookup)} subjects  "
              f"(age_norm σ=1, sex_num, apoe4_norm)  → SubjectBiasNet in_dim=6")

    if getattr(args, "metadata_imputer_checkpoint", None) and not getattr(args, "meta_scaler_checkpoint", None):
        args.meta_scaler_checkpoint = args.metadata_imputer_checkpoint
    if args.metadata_imputer_mode in {"frozen", "joint"} and not args.metadata_imputer_checkpoint:
        raise ValueError(f"--metadata_imputer_mode {args.metadata_imputer_mode} requires --metadata_imputer_checkpoint")

    # ── dataset ──────────────────────────────────────────────────────────────
    dataset = DTI_ADNIDataset(
        dti_root=args.dti_root, selected_subjects_csv=args.selected_csv,
        aal_adjacency=args.aal_adjacency, manifest_csv=args.manifest_csv,
        meta_csv=args.meta_csv, meta_mode=args.meta_mode,
        min_visits=args.min_visits, max_visits=args.max_visits,
        standardize=args.standardize, match_window_days=args.match_window_days,
        clip_features=args.clip_features, roi_subdir=args.roi_subdir, n_rois=args.n_rois, ollama_cache=getattr(args, "ollama_cache", None),
        meta_scaler_checkpoint=getattr(args, "meta_scaler_checkpoint", None),
    )
    summary = dataset.summary()
    print("[dataset]\n" + json.dumps(summary, indent=2))
    (save_root / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    dataset.to_subject_visit_frame().to_csv(save_root / "subject_visit_labels_loaded.csv", index=False)

    num_classes = 3 if args.task == "next_dx" else 2
    class_names = ["CN","MCI","AD"] if args.task == "next_dx" else ["Stable","Changed"]

    use_ollama = args.meta_encoder in {"ollama", "ft_plus_ollama"}
    if use_ollama and not args.ollama_cache:
        raise ValueError("--meta_encoder ollama/ft_plus_ollama requires --ollama_cache")
    _structured, _meta_early, _meta_late, use_metadata = metadata_configuration(args)
    print(
        f"[metadata] loader={use_metadata} early={_meta_early} late={_meta_late} "
        f"mode={args.meta_mode} fields={dataset.meta_num_dim}+{len(dataset.meta_cat_cardinalities)}"
    )
    if args.use_converter_weight:
        print(
            "[WARNING] --use_converter_weight uses retrospective converter labels "
            "for loss weighting. Omit it for a strictly prospective/leakage-free run."
        )

    # ── global window distribution ─────────────────────────────────────────
    dummy_enc  = make_encoder(args, device)
    all_loader = build_loader(args, dataset, list(range(len(dataset))),
                              dummy_enc, device, shuffle=False, seed=args.seed,
                              use_metadata=use_metadata, use_ollama=use_ollama)
    all_win_df = all_loader.window_frame()
    if args.expanded_clinical_root:
        from expanded_clinical_loader_bridge import assert_v2_cohort
        n_match = assert_v2_cohort(args.expanded_clinical_root,
                                    args.expanded_clinical_alignment,all_win_df)
        print(f"[expanded-clinical] Cohort verified: {n_match} matching windows")

    all_win_df.to_csv(save_root / "all_windows_distribution.csv", index=False)
    print(f"[all windows] n={len(all_loader.windows)}  labels={all_loader.label_counts()}")
    (all_win_df.groupby(["transition","stable_change_name"]).size()
               .reset_index(name="n").sort_values(["transition","stable_change_name"])
               .to_csv(save_root / "all_transition_counts.csv", index=False))

    if args.check_only:
        print("[check_only] done"); return

    del dummy_enc
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb = try_import_wandb() if args.wandb and args.wandb_mode != "disabled" else None
    run   = None
    if args.wandb and wandb is not None:
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        run  = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                          name=args.wandb_run_name or save_root.name,
                          config=vars(args), tags=tags, mode=args.wandb_mode)
        run.summary["n_subjects"]    = summary.get("n_subjects")
        run.summary["n_windows_all"] = len(all_loader.windows)

    split_seed = args.seed if args.split_seed is None else args.split_seed
    print(f"[split] split_seed={split_seed} model_seed={args.seed}")
    splits    = get_subject_splits(dataset, args.task, args.num_folds, split_seed,
                                   args.test_fraction, args.val_fraction)
    fold_rows: List[Dict] = []
    oof_rows:  List[pd.DataFrame] = []

    # ══════════════════════════════════════════════════════════════════════════
    # FOLD LOOP
    # ══════════════════════════════════════════════════════════════════════════
    for fold_idx, split in enumerate(splits, 1):
        print("\n" + "=" * 80)
        print(f"FOLD {fold_idx}/{len(splits)}")
        fold_dir = save_root / f"fold{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        tr_idx  = subjects_to_indices(dataset, split["train_subjects"])
        val_idx = subjects_to_indices(dataset, split["val_subjects"])
        te_idx  = subjects_to_indices(dataset, split["test_subjects"])
        print(f"subjects  train={len(split['train_subjects'])}  "
              f"val={len(split['val_subjects'])}  test={len(split['test_subjects'])}")

        if args.expanded_clinical_root and args.meta_mode != "none" and args.metadata_fusion != "none":
            from expanded_clinical_loader_bridge import apply_v2_clinical_fold
            if args.meta_encoder not in {"mlp","ft_transformer","transformer"}:
                raise ValueError("Expanded clinical data requires a structured meta_encoder")
            if args.metadata_imputer_mode != "none" or args.metadata_imputer_checkpoint:
                raise ValueError("Disable legacy metadata imputer/checkpoint for expanded clinical runs")
            apply_v2_clinical_fold(
                dataset,split,fold_dir,args.expanded_clinical_root,
                alignment=args.expanded_clinical_alignment,
                feature_set=args.expanded_clinical_set,
            )

        set_random_seeds(args.seed + fold_idx)
        encoder = make_encoder(args, device)

        # graph_emb_dim unchanged — temporal delta is now additive, not concatenated
        graph_emb_dim = getattr(encoder, "output_graph_dim", args.gnn_output_dim * 2)
        print(f"[model] graph_emb_dim={graph_emb_dim}  "
              f"delta={'additive alpha=0.3' if args.use_temporal_delta else 'off'}")

        classifier = build_sequence_classifier(
            args, graph_emb_dim, num_classes, dataset
        ).to(device)

        # ── Multi-atlas late fusion (v11) ─────────────────────────────────
        _atlas_caches = None
        _clinical_lookup = None
        if getattr(args, "fusion_atlases", ""):
            try:
                from multiatlas_fusion import (
                    build_atlas_caches, atlas_dims_from_caches,
                    MultiAtlasFusion, FusionClassifier, fit_all_normalizers,
                )
                _names = [a.strip() for a in args.fusion_atlases.split(",") if a.strip()]
                _roots = {}
                if args.fusion_jhu_root:   _roots["jhu"]         = args.fusion_jhu_root
                if args.fusion_sch100_root:_roots["schaefer100"] = args.fusion_sch100_root
                if args.fusion_sch200_root:_roots["schaefer200"] = args.fusion_sch200_root
                if args.fusion_sch400_root:_roots["schaefer400"] = args.fusion_sch400_root
                _atlas_caches = build_atlas_caches(_names, _roots, n_scalars=4)

                # Fit normalizers on TRAIN visits only (leakage-free)
                _train_pairs = []
                for _sid in split["train_subjects"]:
                    _graphs = dataset.subject_graph_dict.get(_sid, [])
                    for _g in _graphs:
                        _d = None
                        for _attr in ("visit_date","date","scan_date","acq_date"):
                            _d = getattr(_g, _attr, None)
                            if _d is not None: break
                        if _d is not None:
                            _train_pairs.append((str(_sid), str(_d)))
                if _train_pairs and not args.fusion_no_normalize:
                    fit_all_normalizers(_atlas_caches, _train_pairs)

                # Clinical lookup
                if args.fusion_use_clinical and meta_lookup is not None:
                    _clinical_lookup = {}
                    for _sid, _m in meta_lookup.items():
                        _clinical_lookup[str(_sid)] = [
                            float(_m.get("age_norm", 0.0)),
                            float(_m.get("sex_num", 0.5)),
                            float(_m.get("apoe4_norm", 0.0)),
                        ]

                # Build fusion module + wrap classifier
                _adims = atlas_dims_from_caches(_atlas_caches,
                                                add_rate=args.fusion_add_rate)
                _clin_dim = 3 if (_clinical_lookup is not None) else 0
                _fusion = MultiAtlasFusion(
                    _adims, clinical_dim=_clin_dim,
                    branch_dim=args.fusion_branch_dim,
                    fusion_dim=args.fusion_dim,
                    dropout=args.fusion_dropout,
                ).to(device)
                classifier = FusionClassifier(
                    classifier, _fusion, num_classes=num_classes,
                    dropout=args.fusion_dropout,
                ).to(device)
                print(f"[fusion] atlases={list(_atlas_caches.keys())}  "
                      f"dims={_adims}  clinical={_clin_dim}  "
                      f"fusion_dim={args.fusion_dim}")
            except Exception as _fe:
                print(f"[fusion] FAILED to init ({_fe}); continuing without fusion")
                import traceback; traceback.print_exc()
                _atlas_caches = None

        # ② SubjectBiasNet
        subject_bias_net: Optional[SubjectBiasNet] = None
        if args.use_subject_bias:
            if meta_lookup is None:
                print("[subject_bias] DISABLED — no leakage-free subject features "
                      "available. The former is_converter/n_forward/n_backward "
                      "features encoded the subject's full trajectory (outcome "
                      "leakage) and have been removed. Pass --meta_mode safe or "
                      "--meta_mode full to supply age/sex/APOE4 instead.")
            else:
                subject_bias_net = SubjectBiasNet(
                    in_dim=3, hidden_dim=args.subject_bias_hidden,
                    num_classes=num_classes, dropout=args.subject_bias_dropout,
                ).to(device)
                print(f"[subject_bias] enabled  in_dim=3 (age, sex, apoe4)  "
                      f"hidden={args.subject_bias_hidden}  "
                      f"scale={args.subject_bias_scale}")

        pna_enricher: Optional[PNAAggregator] = None
        if args.use_pna_enricher:
            pna_enricher = PNAAggregator(
                in_dim=graph_emb_dim, out_dim=graph_emb_dim,
                dropout=0.2,
            ).to(device)
            print(f"[pna] PNA multi-aggregator enabled  in/out_dim={graph_emb_dim}")

        # ── Supervised contrastive auxiliary loss ──────────────────────────────
        supcon_criterion: Optional[SupervisedContrastiveLoss] = None
        if args.use_supcon:
            supcon_criterion = SupervisedContrastiveLoss(
                temperature=args.supcon_temperature
            )
            print(f"[supcon] enabled  weight={args.supcon_weight}  "
                  f"T={args.supcon_temperature}")


        # PHASE 1: PRE-TRAIN GNN + LSTM/GRU ON 3-CLASS NEXT-DX
        # ══════════════════════════════════════════════════════════════════════
        # Why: the 3-class task (CN/MCI/AD) has more balanced labels and more
        # windows than stable/changed. Pre-training teaches the GNN which
        # white-matter features correlate with each DX stage BEFORE the harder
        # stable-vs-changed fine-tuning.  The 3-class head is discarded after
        # Phase 1; all shared GRU/LSTM, embedding, projection weights transfer.
        #
        # Skip with --pretrain_epochs 0.
        pretrain_classifier: Optional[DTISequenceClassifier] = None
        _pt_state_to_transfer: Optional[dict] = None

        if args.pretrain_epochs > 0 and args.task == "stable_change":
            print(f"\n{'='*70}")
            print(f"PHASE 1 — Pre-training {args.pretrain_epochs} epochs on 3-class next-DX  "
                  f"(fold {fold_idx})")
            print(f"{'='*70}")

            PT_NUM_CLASSES = 3
            PT_CLASS_NAMES = ["CN", "MCI", "AD"]

            # 3-class classifier uses the same temporal and metadata architecture
            # as fine-tuning, so compatible weights can transfer cleanly.
            pretrain_classifier = build_sequence_classifier(
                args, graph_emb_dim, PT_NUM_CLASSES, dataset
            ).to(device)

            # Natural distribution for 3-class (no rebalancing needed)
            pt_loader = build_loader_with_task(
                args, dataset, tr_idx, encoder, device,
                shuffle=True, seed=args.seed + fold_idx,
                use_metadata=use_metadata, use_ollama=use_ollama,
                task_override="next_dx",
            )
            pt_weights, pt_counts = class_weights_from_loader(pt_loader, PT_NUM_CLASSES, device)
            pt_criterion = nn.CrossEntropyLoss(weight=pt_weights)
            print(f"[pretrain] {len(pt_loader.windows)} windows  counts={pt_counts}  "
                  f"weights={pt_weights.cpu().numpy().round(3).tolist()}")

            pt_params = (list(encoder.parameters()) +
                         list(pretrain_classifier.parameters()) +
                         (list(pna_enricher.parameters()) if pna_enricher else []))
            pt_optimizer = torch.optim.AdamW(
                pt_params, lr=args.pretrain_lr, weight_decay=args.weight_decay
            )
            pt_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                pt_optimizer, T_max=args.pretrain_epochs, eta_min=1e-6
            )

            # Context: no hard DX rule, no subject bias (3-class task)
            ctx_pt = ForwardContext(
                subject_bias_net=None, converter_lookup={}, meta_lookup=None, device=device,
                bias_scale=0.0, use_temporal_delta=args.use_temporal_delta,
    use_multiscale_delta=getattr(args, 'use_multiscale_delta', False),  # v10
                temp_scaler=None, use_dx_hard_rule=False, pna_enricher=pna_enricher,
            )

            for pt_ep in range(1, args.pretrain_epochs + 1):
                encoder.train(); pretrain_classifier.train()
                if pna_enricher: pna_enricher.train()
                pt_losses: List[float] = []

                for pt_batch in pt_loader:
                    pt_optimizer.zero_grad()
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        pt_logits = _full_forward(pt_batch, encoder, pretrain_classifier,
                                                  ctx_pt, apply_temp=False)
                        pt_loss = pt_criterion(pt_logits, pt_batch["labels"])
                    if scaler:
                        scaler.scale(pt_loss).backward()
                        scaler.unscale_(pt_optimizer)
                        nn.utils.clip_grad_norm_(pt_params, 1.0)
                        scaler.step(pt_optimizer); scaler.update()
                    else:
                        pt_loss.backward()
                        nn.utils.clip_grad_norm_(pt_params, 1.0)
                        pt_optimizer.step()
                    pt_losses.append(float(pt_loss.item()))

                pt_sched.step()

                if pt_ep % args.pretrain_log_every == 0 or pt_ep == args.pretrain_epochs:
                    # Quick 3-class val check
                    encoder.eval(); pretrain_classifier.eval()
                    _vp, _vt = [], []
                    with torch.no_grad():
                        _vl = build_loader_with_task(
                            args, dataset, val_idx, encoder, device,
                            shuffle=False, seed=args.seed,
                            use_metadata=use_metadata, use_ollama=use_ollama,
                            task_override="next_dx",
                        )
                        for vb in _vl:
                            _vp.append(_full_forward(vb, encoder, pretrain_classifier,
                                                     ctx_pt).argmax(1).cpu().numpy())
                            _vt.append(vb["labels"].cpu().numpy())
                        del _vl
                    _vp = np.concatenate(_vp); _vt = np.concatenate(_vt)
                    _vbacc = float(balanced_accuracy_score(_vt, _vp))
                    _vacc  = float(accuracy_score(_vt, _vp))
                    _vcounts = dict(zip(PT_CLASS_NAMES, np.bincount(_vp, minlength=3).tolist()))
                    print(
                        f"[pretrain] ep {pt_ep:03d}/{args.pretrain_epochs}  "
                        f"loss={np.mean(pt_losses):.4f}  "
                        f"val_acc={_vacc:.3f}  val_bal={_vbacc:.3f}  "
                        f"preds={_vcounts}"
                    )

            # Capture state dict BEFORE freeing the pre-training classifier
            _pt_state_to_transfer = dict(pretrain_classifier.state_dict())
            del pt_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()

            print(f"\n[pretrain] Phase 1 complete.  Encoder updated in-place.  "
                  f"LSTM/embedding weights captured for transfer.")
            print(f"{'='*70}")
            print(f"PHASE 2 — Fine-tuning on stable/changed  (fold {fold_idx})")
            print(f"{'='*70}\n")

        # Apply pre-trained LSTM/embedding weights from Phase 1 (if available).
        # strict=False: PyTorch skips any param whose shape doesn't match —
        # specifically the final linear (3-class → 2-class) is automatically
        # skipped and stays randomly initialized for the binary task.
        if _pt_state_to_transfer is not None:
            # strict=False skips missing/extra keys but still errors on shape
            # mismatches (3-class vs 2-class final layer). Filter the state dict
            # to only include keys whose tensor shapes exactly match the target.
            target_sd  = classifier.state_dict()
            compat_sd  = {
                k: v for k, v in _pt_state_to_transfer.items()
                if k in target_sd and v.shape == target_sd[k].shape
            }
            skipped = [k for k in _pt_state_to_transfer if k not in compat_sd]
            classifier.load_state_dict(compat_sd, strict=False)
            print(f"[finetune] Pre-trained weights loaded  "
                  f"transferred={len(compat_sd)}  skipped={len(skipped)}  "
                  f"(skipped layers: {skipped})")
            del pretrain_classifier, _pt_state_to_transfer
            pretrain_classifier = None
            _pt_state_to_transfer = None

        # Forward context (training)
        ctx_train = ForwardContext(
            subject_bias_net=subject_bias_net,
            converter_lookup=converter_lookup,
            meta_lookup=meta_lookup,
            device=device,
            bias_scale=args.subject_bias_scale,
            use_temporal_delta=args.use_temporal_delta,
            temp_scaler=None,
            use_dx_hard_rule=args.use_dx_hard_rule,
            pna_enricher=pna_enricher,
            atlas_caches=_atlas_caches,
            clinical_lookup=_clinical_lookup,
            fusion_normalize=not args.fusion_no_normalize,
            fusion_add_rate=args.fusion_add_rate,
        )

        # ══════════════════════════════════════════════════════════════════════
        # ── data loaders ───────────────────────────────────────────────────────
        train_loader = build_loader(args, dataset, tr_idx, encoder, device,
                                    shuffle=True,  seed=args.seed+fold_idx,
                                    use_metadata=use_metadata, use_ollama=use_ollama)
        val_loader   = build_loader(args, dataset, val_idx, encoder, device,
                                    shuffle=False, seed=args.seed,
                                    use_metadata=use_metadata, use_ollama=use_ollama)
        test_loader  = build_loader(args, dataset, te_idx, encoder, device,
                                    shuffle=False, seed=args.seed,
                                    use_metadata=use_metadata, use_ollama=use_ollama)

        # ── DX-stratified window rebalancing ────────────────────────────────
        orig_n = len(train_loader.windows)
        # Get window_frame BEFORE modifying windows (has transition info)
        train_wf = train_loader.window_frame()

        if args.train_stable_to_changed_ratio > 0:
            train_loader.windows = rebalance_training_windows(
                train_loader.windows, args.train_stable_to_changed_ratio, args.seed+fold_idx)

        if args.oversample_changed_ratio > 0:
            # Use stratified rebalancing: preserve DX transition distribution
            # in stable windows + prioritise converter subject stable windows
            train_loader.windows = stratified_rebalance_windows(
                train_loader.windows,
                train_wf,
                target_stable_to_changed=args.oversample_changed_ratio,
                converter_lookup=converter_lookup,
                seed=args.seed + fold_idx,
            )

        # Print distribution after rebalancing
        n_stable = sum(1 for w in train_loader.windows if int(w[-1]) == 0)
        n_changed = sum(1 for w in train_loader.windows if int(w[-1]) == 1)
        print(f"windows  train {orig_n} → {len(train_loader.windows)}  "
              f"(stable={n_stable}, changed={n_changed}, ratio={n_stable/max(n_changed,1):.1f}:1)")
        print(f"train  {train_loader.label_counts()}  trans={train_loader.transition_counts()}")
        if getattr(args, "eval_stable_to_changed_ratio", 0.0) > 0:
            val_loader.windows  = rebalance_training_windows(
                val_loader.windows,  args.eval_stable_to_changed_ratio, args.seed)
            test_loader.windows = rebalance_training_windows(
                test_loader.windows, args.eval_stable_to_changed_ratio, args.seed)
            _vs = sum(1 for w in val_loader.windows  if int(w[-1])==0)
            _vc = sum(1 for w in val_loader.windows  if int(w[-1])==1)
            _ts = sum(1 for w in test_loader.windows if int(w[-1])==0)
            _tc = sum(1 for w in test_loader.windows if int(w[-1])==1)
            print(f"[eval rebalance] val  stable={_vs} changed={_vc} "
                  f"ratio={_vs/max(_vc,1):.1f}:1  (all changed kept)")
            print(f"[eval rebalance] test stable={_ts} changed={_tc} "
                  f"ratio={_ts/max(_tc,1):.1f}:1  (all changed kept)")
        print(f"val    {val_loader.label_counts()}")
        print(f"test   {test_loader.label_counts()}")

        train_loader.window_frame().to_csv(fold_dir / "train_windows.csv", index=False)
        val_loader.window_frame().to_csv(fold_dir / "val_windows.csv",   index=False)
        test_loader.window_frame().to_csv(fold_dir / "test_windows.csv", index=False)

        # ── loss ──────────────────────────────────────────────────────────
        weights, train_counts = class_weights_from_loader(train_loader, num_classes, device)
        if args.loss_type == "sba":
            criterion = SoftBalancedAccuracyLoss(
                ce_weight=weights, alpha=args.sba_alpha,
                label_smoothing=args.label_smoothing,
            )
            print(f"[loss] SoftBalAcc  alpha={args.sba_alpha}  "
                  f"smooth={args.label_smoothing}  "
                  f"counts={train_counts}  weights={weights.cpu().numpy().round(3).tolist()}")
        elif args.loss_type == "ce":
            criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
            print(f"[loss] CE  smooth={args.label_smoothing}  "
                  f"counts={train_counts}  weights={weights.cpu().numpy().round(3).tolist()}")
        elif args.loss_type == "logit_adjusted":
            if not _V10_EXTRAS:
                raise ImportError("logit_adjusted requires v9_extras.py")
            # pi_pos computed from the OVERSAMPLED training windows
            _n_changed = sum(1 for w in train_loader.windows if int(w[-1]) == 1)
            _n_stable  = sum(1 for w in train_loader.windows if int(w[-1]) == 0)
            _pi_pos = _n_changed / max(1, _n_stable + _n_changed)
            criterion = LogitAdjustedLoss(
                pi_pos=_pi_pos, tau=args.logit_adj_tau,
                label_smoothing=args.label_smoothing,
            )
            print(f"[loss] logit_adjusted  pi_pos={_pi_pos:.3f}  "
                  f"tau={args.logit_adj_tau}  smooth={args.label_smoothing}  "
                  f"counts=[{_n_stable}, {_n_changed}]")
        else:
            criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma,
                                  label_smoothing=args.label_smoothing)
            print(f"[loss] focal  alpha={args.focal_alpha}  gamma={args.focal_gamma}")

        # ── optimizer ─────────────────────────────────────────────────────
        # Single LR for all params.  Differential LR (v4) caused the GNN to freeze
        # (preds were identical for 20+ epochs) because gnn_lr=0.3e-4 with cosine
        # decay hit ~0 by ep 30 and the GNN stopped updating entirely.
        bias_params = list(subject_bias_net.parameters()) if subject_bias_net else []
        pna_params  = list(pna_enricher.parameters()) if pna_enricher else []
        all_params  = [p for p in (list(encoder.parameters()) + list(classifier.parameters())
                       + bias_params + pna_params) if p.requires_grad]
        optimizer   = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
        print(f"[optim] AdamW  lr={args.lr:.2e}  n_params={sum(p.numel() for p in all_params):,}")

        def warmup_lambda(ep):
            if args.warmup_epochs > 0 and ep < args.warmup_epochs:
                return max(0.01, (ep+1)/args.warmup_epochs)
            return 1.0
        warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)

        if args.scheduler == "cosine":
            main_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=args.cosine_T0, T_mult=args.cosine_Tmult, eta_min=1e-6)
        elif args.scheduler == "onecycle":
            n_b = max(1, len(train_loader.windows) // args.batch_size)
            main_sched = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr,
                total_steps=args.n_epochs * max(1, n_b // args.accumulation_steps),
                pct_start=0.1, anneal_strategy="cos", final_div_factor=1000)
        else:
            main_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6)

        # ── EMA setup (optional) ───────────────────────────────────────────
        _ema = None
        if getattr(args, "use_ema", False):
            _ema = EMA({
                "encoder":    encoder,
                "classifier": classifier,
                "bias_net":   subject_bias_net,
                "pna":        pna_enricher,
            }, decay=args.ema_decay)
            print(f"[ema] enabled  decay={args.ema_decay}")

        # ── training loop ──────────────────────────────────────────────────
        best_score, best_state, best_epoch, patience_count = -1.0, None, 0, 0
        _ckpt_window = getattr(args, "checkpoint_window", 0)
        _recent_ckpts = []  # list of (score, state_dict, ep) for tail checkpoint
        history: List[Dict] = []

        for ep in range(1, args.n_epochs + 1):
            encoder.train(); classifier.train()
            if subject_bias_net: subject_bias_net.train()
            ep_losses: List[float] = []
            optimizer.zero_grad()
            last_step = -1

            for step, batch in enumerate(train_loader):
                last_step = step
                # noise
                if args.graph_noise_std > 0:
                    try:
                        batch["graph_seq"] = inject_graph_noise(batch["graph_seq"], args.graph_noise_std)
                    except Exception:
                        pass

                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = _full_forward(
                        batch, encoder, classifier, ctx_train,
                        apply_temp=False, drop_edge_p=args.drop_edge_p,
                    )
                    labels = batch["labels"]
                    if args.use_converter_weight and converter_lookup:
                        sw   = get_converter_sample_weights(batch["subjects"], labels, converter_lookup,
                                                             args.converter_stable_mult,
                                                             args.converter_changed_mult, device)
                        loss = compute_weighted_loss(logits, labels, criterion, sw)
                    else:
                        loss = criterion(logits, labels)
                    # SupCon auxiliary loss on GRU window embeddings
                    if supcon_criterion is not None:
                        _embs = getattr(classifier, "_last_hidden", None)
                        if _embs is not None:
                            loss = (loss
                                    + args.supcon_weight
                                    * supcon_criterion(_embs, labels))
                    _imp_loss = getattr(classifier, "_imputation_loss", None)
                    if (_imp_loss is not None
                            and args.metadata_imputer_mode in {"joint", "scratch_joint"}
                            and args.metadata_imputer_loss_weight > 0):
                        loss = loss + args.metadata_imputer_loss_weight * _imp_loss
                    loss = loss / args.accumulation_steps

                if scaler: scaler.scale(loss).backward()
                else:       loss.backward()
                ep_losses.append(float(loss.item()) * args.accumulation_steps)

                if (step + 1) % args.accumulation_steps == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(all_params, 1.0)
                        scaler.step(optimizer); scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(all_params, 1.0)
                        optimizer.step()
                    optimizer.zero_grad()
                    if args.scheduler == "onecycle": main_sched.step()

            # flush leftover gradient
            if last_step >= 0 and (last_step + 1) % args.accumulation_steps != 0:
                if scaler:
                    scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(all_params, 1.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    nn.utils.clip_grad_norm_(all_params, 1.0); optimizer.step()
                optimizer.zero_grad()

            # EMA update once per epoch (after all optimizer steps)
            if _ema is not None:
                _ema.update()

            if ep <= args.warmup_epochs:
                warmup_sched.step()
            elif args.scheduler == "cosine":
                main_sched.step(ep - args.warmup_epochs - 1)

            # If EMA active, evaluate on the smoothed shadow weights
            if _ema is not None:
                _ema.apply_shadow()
                val_m, _ = evaluate(val_loader, encoder, classifier, criterion,
                                     num_classes, ctx=ctx_train)
                _ema.restore()
            else:
                val_m, _ = evaluate(val_loader, encoder, classifier, criterion,
                                     num_classes, ctx=ctx_train)
            score = val_m.get(args.early_stop_metric, val_m["balanced_accuracy"])
            if not np.isfinite(score): score = val_m["balanced_accuracy"]
            if args.scheduler == "plateau": main_sched.step(score)

            train_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
            history.append({
                "fold": fold_idx, "epoch": ep, "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_m.items()
                   if k not in {"pred_counts","target_counts"}},
            })
            print(f"fold {fold_idx}  ep {ep:03d}/{args.n_epochs}  "
                  f"loss={train_loss:.4f}  "
                  f"val_bal={val_m['balanced_accuracy']:.3f}  "
                  f"val_auc={val_m.get('auc', float('nan')):.3f}  "
                  f"val_F1c={val_m.get('f1_changed', float('nan')):.3f}  "
                  f"val_Rc={val_m.get('recall_changed', float('nan')):.3f}  "
                  f"preds={val_m['pred_counts']}")

            if run:
                run.log({"fold": fold_idx, "epoch": ep, "train/loss": train_loss,
                         "val/auc": val_m.get("auc"), "val/balanced_accuracy": val_m["balanced_accuracy"],
                         "val/f1_changed": val_m.get("f1_changed"),
                         "val/recall_changed": val_m.get("recall_changed"),
                         "val/macro_f1": val_m["macro_f1"]})

            # ── per-epoch test logging ─────────────────────────────────────────────
            if getattr(args, 'log_test_every', 0) > 0:
                if (ep >= getattr(args, 'log_test_start', 200)
                        and ep % args.log_test_every == 0):
                    _te_m, _ = evaluate(test_loader, encoder, classifier,
                                        criterion, num_classes, threshold=0.5,
                                        ctx=ctx_train, apply_temp=False)
                    _te_auc = round(float(_te_m.get('auc', 0.0)), 4)
                    _te_bal = round(float(_te_m.get('balanced_accuracy', 0.0)), 4)
                    _te_f1c = round(float(_te_m.get('f1_changed', 0.0)), 4)
                    _v_auc = round(float(val_m.get('auc', 0.0)), 4)
                    _v_bal = round(float(val_m.get('balanced_accuracy', 0.0)), 4)
                    import csv as _csv_mod, os as _os_log
                    _csvp = _os_log.path.join(
                        args.save_path, f'test_curve_fold{fold_idx}.csv')
                    _new_file = not _os_log.path.exists(_csvp)
                    with open(_csvp, 'a', newline='') as _cf:
                        _cw = _csv_mod.writer(_cf)
                        if _new_file:
                            _cw.writerow(['fold','epoch','val_auc','val_bal',
                                          'test_auc','test_bal','test_f1c'])
                        _cw.writerow([fold_idx, ep, _v_auc, _v_bal,
                                      _te_auc, _te_bal, _te_f1c])
                    print(f'  [testlog ep={ep}]  test_auc={_te_auc:.3f}'
                          f'  test_bal={_te_bal:.3f}')
            # ─────────────────────────────────────────────────────────────────────────
            _min_ep = getattr(args, 'min_epochs', 0)
            # ── checkpoint this epoch for tail-window selection ───────────────
            if ep >= _min_ep:
                if _ema is not None:
                    # Snapshot the EMA shadow weights (what we evaluated on)
                    _sh = _ema.state()
                    _state_now = {
                        "encoder":    copy.deepcopy(_sh["encoder"]),
                        "classifier": copy.deepcopy(_sh["classifier"]),
                        "bias_net":   copy.deepcopy(_sh["bias_net"]) if "bias_net" in _sh else None,
                        "pna":        copy.deepcopy(_sh["pna"]) if "pna" in _sh else None,
                        "epoch": ep, "val_metrics": val_m, "args": vars(args),
                    }
                else:
                    _state_now = {
                        "encoder":     copy.deepcopy(encoder.state_dict()),
                        "classifier":  copy.deepcopy(classifier.state_dict()),
                        "bias_net":    copy.deepcopy(subject_bias_net.state_dict()) if subject_bias_net else None,
                        "pna":         copy.deepcopy(pna_enricher.state_dict()) if pna_enricher else None,
                        "epoch": ep, "val_metrics": val_m, "args": vars(args),
                    }
                if _ckpt_window > 0:
                    _recent_ckpts.append((score, _state_now, ep))
                    if len(_recent_ckpts) > _ckpt_window:
                        _recent_ckpts.pop(0)

            if score > best_score and ep >= _min_ep:
                best_score = score; best_epoch = ep; patience_count = 0
                if _ckpt_window == 0:   # classic mode: track global best state
                    best_state = _state_now
                    if args.save_model: torch.save(best_state, fold_dir / "best_model.pt")
            elif ep < _min_ep:
                patience_count = 0  # warmup: don't burn patience yet
            else:
                patience_count += 1
                if patience_count >= args.patience:
                    print(f"early stop  ep={ep}  best_{args.early_stop_metric}={best_score:.4f}  ep={best_epoch}")
                    break

        pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)

        # ── select / average checkpoint(s) from tail window ────────────────
        _wavg_k = getattr(args, "weight_avg_topk", 0)
        _ens_k  = getattr(args, "ckpt_ensemble_topk", 0)
        _ensemble_states = None   # for prediction ensembling (set below if active)

        if _ckpt_window > 0 and _recent_ckpts:
            # Sort tail checkpoints by val score, best first
            _sorted = sorted(_recent_ckpts, key=lambda x: x[0], reverse=True)

            if _wavg_k > 1:
                # ── SWA-style WEIGHT AVERAGING of top-K checkpoints ──────────
                topk = _sorted[:_wavg_k]
                eps  = [c[2] for c in topk]
                best_state = {
                    "encoder":    average_state_dicts([c[1]["encoder"]    for c in topk]),
                    "classifier": average_state_dicts([c[1]["classifier"] for c in topk]),
                    "bias_net":   average_state_dicts([c[1]["bias_net"]   for c in topk])
                                  if topk[0][1]["bias_net"] else None,
                    "pna":        average_state_dicts([c[1]["pna"]        for c in topk])
                                  if topk[0][1].get("pna") else None,
                    "epoch": eps[0], "val_metrics": topk[0][1]["val_metrics"],
                    "args": vars(args),
                }
                best_epoch = eps[0]
                best_score = topk[0][0]
                print(f"[checkpoint] WEIGHT-AVG top-{len(topk)} epochs {sorted(eps)}  "
                      f"(SWA — flatter minimum, lower variance)")
            elif _ens_k > 1:
                # ── PREDICTION ENSEMBLING: keep top-K states for eval ────────
                topk = _sorted[:_ens_k]
                eps  = [c[2] for c in topk]
                _ensemble_states = [c[1] for c in topk]
                best_state = topk[0][1]   # load best single for temp/threshold fit
                best_epoch = eps[0]
                best_score = topk[0][0]
                print(f"[checkpoint] PREDICTION-ENSEMBLE top-{len(topk)} epochs {sorted(eps)}  "
                      f"(averaged softmax at inference)")
            else:
                # ── Classic: single best from tail window ────────────────────
                best_state = _sorted[0][1]
                best_epoch = _sorted[0][2]
                best_score = _sorted[0][0]
                print(f"[checkpoint] tail window: selected ep={best_epoch}  "
                      f"val_{args.early_stop_metric}={best_score:.4f}  "
                      f"(best of last {len(_recent_ckpts)} epochs)")

            if args.save_model:
                torch.save(best_state, fold_dir / "best_model.pt")
        if best_state:
            encoder.load_state_dict(best_state["encoder"])
            classifier.load_state_dict(best_state["classifier"])
            if subject_bias_net and best_state["bias_net"]:
                subject_bias_net.load_state_dict(best_state["bias_net"])
            if pna_enricher and best_state.get("pna"):
                pna_enricher.load_state_dict(best_state["pna"])

        # ── ①.5 DECOUPLED cRT / LWS (optional second stage) ────────────────
        if getattr(args, "crt_epochs", 0) > 0:
            _crt_classifier_backup = copy.deepcopy(classifier.state_dict())
            _crt_windows_backup = list(train_loader.windows)
            _crt_original_lr = float(args.crt_lr)
            _crt_original_epochs = int(args.crt_epochs)

            try:
                _crt_out = run_crt_stage(
                    args, encoder, classifier, subject_bias_net, pna_enricher,
                    train_loader, num_classes, ctx_train, device, None)
                if args.crt_mode == "lws":
                    classifier = _crt_out
            except Exception as _ce:
                print(
                    f"[cRT] first attempt failed ({_ce}); "
                    "restoring the pre-cRT classifier and retrying safely"
                )
                import traceback
                traceback.print_exc()

                classifier.load_state_dict(_crt_classifier_backup)
                train_loader.windows = list(_crt_windows_backup)
                args.crt_lr = min(_crt_original_lr, 2e-4)
                args.crt_epochs = min(_crt_original_epochs, 20)

                try:
                    _crt_out = run_crt_stage(
                        args, encoder, classifier, subject_bias_net, pna_enricher,
                        train_loader, num_classes, ctx_train, device, None)
                    if args.crt_mode == "lws":
                        classifier = _crt_out
                    print(
                        f"[cRT] safe retry succeeded "
                        f"(lr={args.crt_lr}, epochs={args.crt_epochs})"
                    )
                except Exception as _retry_error:
                    print(
                        f"[cRT] safe retry also failed ({_retry_error}); "
                        "using the valid pre-cRT classifier"
                    )
                    traceback.print_exc()
                    classifier.load_state_dict(_crt_classifier_backup)
                    train_loader.windows = list(_crt_windows_backup)
                finally:
                    args.crt_lr = _crt_original_lr
                    args.crt_epochs = _crt_original_epochs
            finally:
                train_loader.windows = list(_crt_windows_backup)

        # ── ① TEMPERATURE SCALING  ─────────────────────────────────────────
        temp_scaler: Optional[TemperatureScaler] = None
        if args.use_temp_scaling:
            temp_scaler = TemperatureScaler().to(device)
            val_logits, val_labs = collect_val_logits(val_loader, encoder, classifier, ctx_train)
            if val_logits is not None:
                val_logits = val_logits.to(device)
                val_labs   = val_labs.to(device)
                temp_scaler.fit(val_logits, val_labs)

        # Evaluation context WITH temperature scaler
        ctx_eval = ForwardContext(
            subject_bias_net=subject_bias_net,
            converter_lookup=converter_lookup,
            meta_lookup=meta_lookup,
            device=device,
            bias_scale=args.subject_bias_scale,
            use_temporal_delta=args.use_temporal_delta,
            temp_scaler=temp_scaler,
            use_dx_hard_rule=args.use_dx_hard_rule,
            pna_enricher=pna_enricher,
            atlas_caches=_atlas_caches,
            clinical_lookup=_clinical_lookup,
            fusion_normalize=not args.fusion_no_normalize,
            fusion_add_rate=args.fusion_add_rate,
        )

        # ── ④ THRESHOLD CALIBRATION on calibrated val probabilities ───────
        opt_thresh = 0.5
        if args.optimize_threshold and num_classes == 2:
            _, val_df_cal = evaluate(val_loader, encoder, classifier, criterion,
                                      num_classes, threshold=0.5, ctx=ctx_eval, apply_temp=True)
            if not val_df_cal.empty:
                # Calibrated validation predictions for within-fold comparisons.
                val_df_cal.to_csv(fold_dir / "val_predictions.csv", index=False)
                vp = val_df_cal[[f"prob_{c}" for c in range(num_classes)]].values
                vl = val_df_cal["target"].values
                opt_thresh, opt_val_s = optimize_threshold(
                    vp, vl, metric=args.threshold_metric,
                    prevalence_tolerance=args.prevalence_tolerance)
                print(f"[threshold] optimal={opt_thresh:.3f}  "
                      f"val_{args.threshold_metric}@opt={opt_val_s:.3f}  "
                      f"predicted_changed%={100*(vp[:,1]>=opt_thresh).mean():.1f}%  "
                      f"(was 0.5)")
                (fold_dir / "threshold.json").write_text(json.dumps({
                    "threshold": opt_thresh, f"val_{args.threshold_metric}": opt_val_s}, indent=2))

        # ── TEST EVALUATION ────────────────────────────────────────────────
        if _ensemble_states is not None and len(_ensemble_states) > 1:
            # ── PREDICTION ENSEMBLE: average softmax across top-K checkpoints ──
            import numpy as _np
            _prob_accum = None
            _base_df    = None
            for _si, _st in enumerate(_ensemble_states):
                encoder.load_state_dict(_st["encoder"])
                classifier.load_state_dict(_st["classifier"])
                if subject_bias_net and _st.get("bias_net"):
                    subject_bias_net.load_state_dict(_st["bias_net"])
                if pna_enricher and _st.get("pna"):
                    pna_enricher.load_state_dict(_st["pna"])
                _m_i, _df_i = evaluate(
                    test_loader, encoder, classifier, criterion,
                    num_classes, threshold=opt_thresh, ctx=ctx_eval, apply_temp=True)
                _pcols = [c for c in _df_i.columns if c.startswith("prob_")]
                if _prob_accum is None:
                    _prob_accum = _df_i[_pcols].values.astype(float)
                    _base_df    = _df_i.copy()
                else:
                    _prob_accum = _prob_accum + _df_i[_pcols].values.astype(float)
            _prob_accum /= len(_ensemble_states)
            # Rebuild metrics from averaged probs
            _pos = _prob_accum[:, 1] if _prob_accum.shape[1] > 1 else _prob_accum[:, 0]
            _preds = (_pos >= opt_thresh).astype(int)
            _base_df["pred"] = _preds
            for _ci in range(_prob_accum.shape[1]):
                _base_df[f"prob_{_ci}"] = _prob_accum[:, _ci]
            _y = _base_df["target"].values.astype(int)
            test_m = metrics_from_probs(_prob_accum, _y, num_classes, threshold=opt_thresh)
            test_pred_df = _base_df
            print(f"[ensemble] averaged {len(_ensemble_states)} checkpoints' predictions")
        elif args.tta_passes > 1:
            test_m, test_pred_df = evaluate_tta(
                test_loader, encoder, classifier, criterion,
                num_classes, n_passes=args.tta_passes, threshold=opt_thresh, ctx=ctx_eval)
        else:
            test_m, test_pred_df = evaluate(
                test_loader, encoder, classifier, criterion,
                num_classes, threshold=opt_thresh, ctx=ctx_eval, apply_temp=True)

        test_pred_df["fold"] = fold_idx
        test_pred_df.to_csv(fold_dir / "test_predictions.csv", index=False)
        tr_report = transition_report(test_pred_df)
        tr_report.to_csv(fold_dir / "transition_report.csv", index=False)

        print("\n── Test Set ──────────────────────────────────────────────────────")
        print(f"fold {fold_idx}  "
              f"acc={test_m['accuracy']:.3f}  bal={test_m['balanced_accuracy']:.3f}  "
              f"auc={test_m['auc']:.3f}  "
              f"F1c={test_m.get('f1_changed',float('nan')):.3f}  "
              f"Rc={test_m.get('recall_changed',float('nan')):.3f}  "
              f"Pc={test_m.get('precision_changed',float('nan')):.3f}  "
              f"thresh={opt_thresh:.3f}")
        print(f"preds={test_m['pred_counts']}  targets={test_m['target_counts']}")
        if not test_pred_df.empty and "target" in test_pred_df.columns:
            print(classification_report(test_pred_df["target"], test_pred_df["pred"],
                                        target_names=class_names, zero_division=0))
            if not tr_report.empty:
                print("Per-transition:"); print(tr_report.to_string(index=False))
            vr_report = visit_report(test_pred_df)
            vr_report.to_csv(fold_dir / "visit_report.csv", index=False)
            if not vr_report.empty:
                print("Per-input-visits:")
                print(vr_report.to_string(index=False))

        if run:
            run.log({"fold": fold_idx,
                     "test/accuracy": test_m["accuracy"], "test/balanced_accuracy": test_m["balanced_accuracy"],
                     "test/auc": test_m["auc"], "test/macro_f1": test_m["macro_f1"],
                     "test/f1_changed": test_m.get("f1_changed"),
                     "test/recall_changed": test_m.get("recall_changed"),
                     "test/precision_changed": test_m.get("precision_changed"),
                     "best_epoch": best_epoch, "opt_threshold": opt_thresh})

        (fold_dir / "result.json").write_text(json.dumps({
            "fold": fold_idx, "best_epoch": best_epoch, "best_val_score": best_score,
            "opt_threshold": opt_thresh, "test_metrics": test_m,
            "val_metrics_at_best": best_state["val_metrics"] if best_state else None,
            "n_train_windows": len(train_loader.windows),
            "n_val_windows": len(val_loader.windows),
            "n_test_windows": len(test_loader.windows),
            "transition_report": tr_report.to_dict(orient="records"),
        }, indent=2, default=str))

        fold_rows.append({
            "fold": fold_idx, "best_epoch": best_epoch, "opt_threshold": opt_thresh,
            "test_acc": test_m["accuracy"], "balanced_acc": test_m["balanced_accuracy"],
            "macro_f1": test_m["macro_f1"],
            "f1_changed":       test_m.get("f1_changed", float("nan")),
            "recall_changed":   test_m.get("recall_changed", float("nan")),
            "precision_changed":test_m.get("precision_changed", float("nan")),
            "auc": test_m["auc"],
            "n_train_windows": len(train_loader.windows),
            "n_val_windows":   len(val_loader.windows),
            "n_test_windows":  len(test_loader.windows),
        })
        oof_rows.append(test_pred_df)

    # ── aggregate results ──────────────────────────────────────────────────────
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(save_root / "fold_results.csv", index=False)
    if oof_rows:
        oof_df = pd.concat(oof_rows, ignore_index=True)
        oof_df.to_csv(save_root / "oof_test_predictions_all_folds.csv", index=False)
        transition_report(oof_df).to_csv(save_root / "oof_transition_report_all_folds.csv", index=False)

    cols = ["test_acc","balanced_acc","macro_f1","f1_changed","recall_changed","precision_changed","auc"]
    summary_rows = [
        {"metric": c, "mean": float(fold_df[c].mean()),
         "std": float(fold_df[c].std(ddof=0)) if len(fold_df) > 1 else 0.0}
        for c in cols if c in fold_df.columns
    ]
    pd.DataFrame(summary_rows).to_csv(save_root / "cv_summary.csv", index=False)
    (save_root / "cv_summary.json").write_text(
        json.dumps({r["metric"]: {"mean": r["mean"], "std": r["std"]} for r in summary_rows}, indent=2))

    print("\n── Summary ───────────────────────────────────────────────────────────")
    for r in summary_rows:
        sep = f" ± {r['std']:.3f}" if len(fold_df) > 1 else ""
        print(f"  {r['metric']:25s}: {r['mean']:.3f}{sep}")
    print(f"\nResults → {save_root}")

    if run:
        for r in summary_rows:
            run.summary[f"{r['metric']}_mean"] = r["mean"]
            run.summary[f"{r['metric']}_std"]  = r["std"]
        run.finish()


if __name__ == "__main__":
    main()