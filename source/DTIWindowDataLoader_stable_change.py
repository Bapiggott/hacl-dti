#!/usr/bin/env python3
"""Window loader for stable-vs-changed forecasting with optional tract edges.

The loader supports three graph modes while preserving one batched code path:

1. Static graph
   Every visit uses the adjacency loaded by :class:`DTI_ADNIDataset`.

2. Dynamic Top-K tract graph
   Every visit constructs its own Top-K support and edge weights from
   ``tract_root/<subject>_<date>/aal116_tract_counts_log_norm.npy``.

3. Fixed support + visit-specific weights
   Every visit uses the same support from ``fixed_tract_support`` but obtains
   its edge weights from that visit's tract matrix. This is the recommended
   compromise for directly comparable ROIs and subject-specific connectivity.

All modes support metadata, Ollama embeddings, previous diagnosis, TopK node
pooling, and weighted GraphSAGE because graph objects are batched together
before the encoder is called.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data

from temporal_gap_processor import normalize_time_gaps

DX_NAMES = {0: "CN", 1: "MCI", 2: "AD"}


def _clean_symmetric_matrix(path: Path, expected_nodes: int) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.shape != (expected_nodes, expected_nodes):
        raise ValueError(
            f"Tract/support matrix {path} has shape {arr.shape}; "
            f"expected {(expected_nodes, expected_nodes)}"
        )
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = 0.5 * (arr + arr.T)
    np.fill_diagonal(arr, 0.0)
    return arr


class DTIWindowDataLoaderStableChange:
    def __init__(
        self,
        dataset,
        subject_indices,
        encoder,
        device,
        batch_size=8,
        shuffle=True,
        seed=42,
        task="stable_change",
        window_mode="sliding",
        max_input_visits=9,
        use_prev_dx=False,
        use_metadata=False,
        use_ollama=False,
        use_time_features=False,
        time_normalization="log",
        drop_reverse_windows=False,
        split_at_reverse_windows=False,
        dynamic_tract_edges=False,
        tract_root=None,
        tract_topk=20,
        fixed_tract_support=None,
        tract_missing_policy="skip_window",
    ):
        self.dataset = dataset
        self.subject_indices = list(subject_indices)
        self.encoder = encoder
        self.device = device
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.task = str(task)
        self.window_mode = str(window_mode)
        self.max_input_visits = int(max_input_visits) if max_input_visits else 0
        self.use_prev_dx = bool(use_prev_dx)
        self.use_metadata = bool(use_metadata)
        self.use_ollama = bool(use_ollama)
        self.use_time_features = bool(use_time_features)
        self.time_normalization = str(time_normalization)
        self.drop_reverse_windows = bool(drop_reverse_windows)
        self.split_at_reverse_windows = bool(split_at_reverse_windows)

        self.dynamic_tract_edges = bool(dynamic_tract_edges)
        self.tract_root = Path(tract_root).expanduser() if tract_root else None
        self.tract_topk = int(tract_topk)
        self.fixed_tract_support_path = (
            Path(fixed_tract_support).expanduser() if fixed_tract_support else None
        )
        self.tract_missing_policy = str(tract_missing_policy)
        if self.tract_missing_policy not in {"skip_window", "static_fallback", "error"}:
            raise ValueError(
                "tract_missing_policy must be skip_window, static_fallback, or error"
            )
        if self.dynamic_tract_edges and self.tract_root is None:
            raise ValueError("--dynamic_tract_edges requires --tract_root")
        if self.fixed_tract_support_path is not None and not self.dynamic_tract_edges:
            raise ValueError(
                "--fixed_tract_support requires --dynamic_tract_edges because the "
                "visit-specific tract matrices supply the edge weights"
            )
        if self.fixed_tract_support_path is not None and not self.fixed_tract_support_path.exists():
            raise FileNotFoundError(
                f"Fixed tract support does not exist: {self.fixed_tract_support_path}"
            )

        self.n_rois = int(getattr(dataset, "n_rois", 116))
        self._tract_cache: Dict[str, np.ndarray] = {}
        self._edge_cache: Dict[Tuple[str, int, str], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._fixed_edge_index_cpu: Optional[torch.Tensor] = None
        if self.fixed_tract_support_path is not None:
            support = _clean_symmetric_matrix(
                self.fixed_tract_support_path, self.n_rois
            ) > 0
            src, dst = np.nonzero(support)
            self._fixed_edge_index_cpu = torch.tensor(
                np.stack([src, dst], axis=0), dtype=torch.long
            )
            if self._fixed_edge_index_cpu.size(1) == 0:
                raise ValueError("Fixed tract support contains no edges")

        self.subject_data = self._group_by_subject()
        self.windows = self._build_windows()

    def _group_by_subject(self) -> Dict[str, List[int]]:
        subject_groups: Dict[str, List[int]] = {}
        for idx in self.subject_indices:
            data = self.dataset[idx]
            sid = getattr(data, "subject_id", None)
            if sid is None:
                sid = getattr(data, "subj_id", None)
            if sid is None:
                continue
            sid = str(sid)
            if "_run" in sid:
                sid = sid.split("_run")[0]
            subject_groups.setdefault(sid, []).append(idx)
        for sid, indices in subject_groups.items():
            indices.sort(
                key=lambda i: int(
                    getattr(
                        self.dataset[i], "visit_order", torch.tensor([0])
                    ).view(-1)[0].item()
                )
            )
        return subject_groups

    def _transition_info(self, input_indices: List[int], target_idx: int):
        prev = self.dataset[input_indices[-1]]
        target = self.dataset[target_idx]
        prev_dx = int(prev.dx_label.view(-1)[0].item())
        target_dx = int(target.dx_label.view(-1)[0].item())
        transition = f"{DX_NAMES.get(prev_dx, prev_dx)}->{DX_NAMES.get(target_dx, target_dx)}"
        is_reverse = target_dx < prev_dx
        is_changed = int(target_dx != prev_dx)
        return prev_dx, target_dx, transition, is_reverse, is_changed

    def _window_label(self, input_indices: List[int], target_idx: int) -> Optional[int]:
        target = self.dataset[target_idx]
        if self.task == "next_dx":
            return int(target.dx_label.view(-1)[0].item())
        if self.task == "converter":
            return int(target.converter_label.view(-1)[0].item())
        if self.task == "progression":
            prev_dx, target_dx, *_ = self._transition_info(input_indices, target_idx)
            if target_dx < prev_dx and self.drop_reverse_windows:
                return None
            return 1 if target_dx > prev_dx else 0
        if self.task in {"stable_change", "stability", "changed"}:
            _prev, _target, _transition, is_reverse, is_changed = self._transition_info(
                input_indices, target_idx
            )
            if is_reverse and self.drop_reverse_windows:
                return None
            return int(is_changed)
        raise ValueError(f"Unknown task: {self.task}")

    def _visit_id(self, data: Data) -> str:
        if hasattr(data, "visit_id"):
            return str(data.visit_id)
        sid = str(getattr(data, "subject_id"))
        date = str(getattr(data, "visit_date"))[:10]
        return f"{sid}_{date}"

    def _tract_path(self, data: Data) -> Path:
        assert self.tract_root is not None
        return self.tract_root / self._visit_id(data) / "aal116_tract_counts_log_norm.npy"

    def _tract_exists(self, idx: int) -> bool:
        if not self.dynamic_tract_edges:
            return True
        return self._tract_path(self.dataset[idx]).exists()

    def _build_windows(self) -> List[Tuple[str, str, List[int], int, int]]:
        windows: List[Tuple[str, str, List[int], int, int]] = []
        for subject, indices in self.subject_data.items():
            if len(indices) < 2:
                continue

            if self.split_at_reverse_windows:
                segments: List[List[int]] = []
                current = [indices[0]]
                for idx in indices[1:]:
                    prev_dx = int(self.dataset[current[-1]].dx_label.view(-1)[0].item())
                    curr_dx = int(self.dataset[idx].dx_label.view(-1)[0].item())
                    if curr_dx < prev_dx:
                        if len(current) >= 2:
                            segments.append(current)
                        current = [idx]
                    else:
                        current.append(idx)
                if len(current) >= 2:
                    segments.append(current)
            else:
                segments = [indices]

            for segment_number, segment_indices in enumerate(segments, start=1):
                if len(segment_indices) < 2:
                    continue
                segment_name = (
                    f"{subject}__seg{segment_number:02d}"
                    if self.split_at_reverse_windows
                    else subject
                )
                target_positions = (
                    [len(segment_indices) - 1]
                    if self.window_mode == "final_only"
                    else list(range(1, len(segment_indices)))
                )
                for target_position in target_positions:
                    start = 0
                    if self.max_input_visits and target_position > self.max_input_visits:
                        start = target_position - self.max_input_visits
                    input_indices = segment_indices[start:target_position]
                    target_idx = segment_indices[target_position]
                    if not input_indices:
                        continue
                    label = self._window_label(input_indices, target_idx)
                    if label is None:
                        continue

                    if self.dynamic_tract_edges and self.tract_missing_policy == "skip_window":
                        if not all(self._tract_exists(i) for i in input_indices):
                            continue
                    windows.append(
                        (segment_name, subject, input_indices, target_idx, int(label))
                    )
        return windows

    def __len__(self):
        return (len(self.windows) + self.batch_size - 1) // self.batch_size

    def label_counts(self) -> Dict[int, int]:
        return dict(Counter(label for *_rest, label in self.windows))

    def transition_counts(self) -> Dict[str, int]:
        counts = Counter()
        for _segment, _subject, input_indices, target_idx, _label in self.windows:
            *_unused, transition, _reverse, _changed = self._transition_info(
                input_indices, target_idx
            )
            counts[transition] += 1
        return dict(sorted(counts.items()))

    def window_frame(self):
        rows = []
        for segment, subject, input_indices, target_idx, label in self.windows:
            prev_dx, target_dx, transition, is_reverse, _changed = self._transition_info(
                input_indices, target_idx
            )
            target = self.dataset[target_idx]
            rows.append(
                {
                    "subject_id": subject,
                    "segment_id": segment,
                    "n_input_visits": len(input_indices),
                    "input_dates": ";".join(
                        [str(self.dataset[i].visit_date) for i in input_indices]
                    ),
                    "target_date": str(target.visit_date),
                    "prev_dx": prev_dx,
                    "prev_dx_name": DX_NAMES.get(prev_dx, str(prev_dx)),
                    "target_dx": target_dx,
                    "target_dx_name": DX_NAMES.get(target_dx, str(target_dx)),
                    "transition": transition,
                    "is_reverse": bool(is_reverse),
                    "stable_change_label": int(label),
                    "stable_change_name": "changed" if int(label) == 1 else "stable",
                    "dynamic_tract_edges": self.dynamic_tract_edges,
                    "fixed_tract_support": str(self.fixed_tract_support_path or ""),
                }
            )
        return pd.DataFrame(rows)

    def _load_tract_matrix(self, data: Data) -> np.ndarray:
        visit_id = self._visit_id(data)
        if visit_id not in self._tract_cache:
            path = self._tract_path(data)
            if not path.exists():
                raise FileNotFoundError(f"Missing tract matrix: {path}")
            self._tract_cache[visit_id] = _clean_symmetric_matrix(path, self.n_rois)
        return self._tract_cache[visit_id]

    def _dynamic_edges(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        visit_id = self._visit_id(data)
        mode = "fixed_support" if self._fixed_edge_index_cpu is not None else "dynamic_topk"
        key = (visit_id, self.tract_topk, mode)
        if key in self._edge_cache:
            return self._edge_cache[key]

        matrix = self._load_tract_matrix(data)
        if self._fixed_edge_index_cpu is not None:
            edge_index = self._fixed_edge_index_cpu.clone()
            src = edge_index[0].numpy()
            dst = edge_index[1].numpy()
            weights = matrix[src, dst]
        else:
            support = np.zeros_like(matrix, dtype=bool)
            for roi in range(matrix.shape[0]):
                row = matrix[roi].copy()
                row[roi] = 0.0
                positive = np.flatnonzero(row > 0)
                if positive.size == 0:
                    continue
                k = min(self.tract_topk, int(positive.size))
                chosen = positive[np.argsort(row[positive])[-k:]]
                support[roi, chosen] = True
            support = np.logical_or(support, support.T)
            src, dst = np.nonzero(support)
            edge_index = torch.tensor(
                np.stack([src, dst], axis=0), dtype=torch.long
            )
            weights = matrix[src, dst]

        edge_weight = torch.tensor(weights, dtype=torch.float32)
        self._edge_cache[key] = (edge_index, edge_weight)
        return edge_index, edge_weight

    def _prepare_graph(self, idx: int) -> Data:
        source = self.dataset[idx]
        graph = source.clone()
        graph._innovation_real_tract = False
        if self.dynamic_tract_edges:
            try:
                edge_index, edge_weight = self._dynamic_edges(source)
                graph._innovation_real_tract = True
                graph.edge_index = edge_index
                graph.edge_weight = edge_weight
            except FileNotFoundError:
                if self.tract_missing_policy == "error":
                    raise
                if self.tract_missing_policy == "static_fallback":
                    graph.edge_index = source.edge_index.clone()
                    graph.edge_weight = getattr(
                        source,
                        "edge_weight",
                        torch.ones(source.edge_index.size(1), dtype=torch.float32),
                    ).clone()
                else:
                    raise
        else:
            graph.edge_index = source.edge_index.clone()
            graph.edge_weight = getattr(
                source,
                "edge_weight",
                torch.ones(source.edge_index.size(1), dtype=torch.float32),
            ).clone()
        return graph

    def _pad_graph_sequences(self, sequences):
        lengths = torch.tensor(
            [seq.size(0) for seq in sequences], dtype=torch.long, device=self.device
        )
        max_len = int(lengths.max().item())
        dim = sequences[0].size(1)
        padded = torch.zeros(len(sequences), max_len, dim, device=self.device)
        for i, seq in enumerate(sequences):
            padded[i, : seq.size(0)] = seq
        return padded, lengths

    def _pad_visit_tensors(self, tensors, pad_value=0, dtype=torch.long):
        max_len = max(t.size(0) for t in tensors)
        if tensors[0].dim() == 1:
            out = torch.full(
                (len(tensors), max_len),
                pad_value,
                dtype=dtype,
                device=self.device,
            )
            for i, tensor in enumerate(tensors):
                out[i, : tensor.size(0)] = tensor.to(self.device)
        else:
            feat_dim = tensors[0].size(1)
            out = torch.zeros(
                (len(tensors), max_len, feat_dim),
                dtype=dtype,
                device=self.device,
            )
            for i, tensor in enumerate(tensors):
                out[i, : tensor.size(0)] = tensor.to(self.device)
        return out

    def __iter__(self):
        order = np.arange(len(self.windows))
        if self.shuffle:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(order)

        for start in range(0, len(order), self.batch_size):
            batch_ids = order[start : start + self.batch_size]
            batch_windows = [self.windows[int(i)] for i in batch_ids]

            all_data = []
            visit_to_window = []
            labels = []
            subjects = []
            segments = []
            target_dates = []
            input_dates = []
            time_gaps = []
            transitions = []
            prev_dx_labels = []
            target_dx_labels = []

            prev_dx_seq_raw = []
            meta_num_seq_raw = []
            meta_mask_seq_raw = []
            meta_cat_seq_raw = []
            meta_present_seq_raw = []
            meta_age_days_seq_raw = []
            meta_baseline_num_seq_raw = []
            meta_baseline_mask_seq_raw = []
            meta_months_since_baseline_seq_raw = []
            meta_months_since_previous_seq_raw = []
            ollama_seq_raw = []
            ollama_present_seq_raw = []

            for window_index, (
                segment,
                subject,
                input_indices,
                target_idx,
                label,
            ) in enumerate(batch_windows):
                labels.append(label)
                subjects.append(subject)
                segments.append(segment)
                target = self.dataset[target_idx]
                target_dates.append(str(target.visit_date))
                input_dates.append(
                    ";".join([str(self.dataset[i].visit_date) for i in input_indices])
                )

                prev_dx, target_dx, transition, _reverse, _changed = self._transition_info(
                    input_indices, target_idx
                )
                transitions.append(transition)
                prev_dx_labels.append(prev_dx)
                target_dx_labels.append(target_dx)

                last_input = self.dataset[input_indices[-1]]
                gap = max(
                    float(target.visit_months.view(-1)[0].item())
                    - float(last_input.visit_months.view(-1)[0].item()),
                    0.0,
                )
                time_gaps.append(gap)

                dx_values = []
                meta_nums = []
                meta_masks = []
                meta_cats = []
                meta_presents = []
                meta_age_days = []
                meta_baseline_nums = []
                meta_baseline_masks = []
                meta_months_since_baselines = []
                meta_months_since_previous = []
                previous_visit_month = None
                ollama_embeddings = []
                ollama_presents = []

                for idx in input_indices:
                    data = self._prepare_graph(idx)
                    all_data.append(data)
                    visit_to_window.append(window_index)
                    dx_values.append(int(data.dx_label.view(-1)[0].item()))

                    if self.use_metadata:
                        meta_nums.append(data.meta_num)
                        meta_masks.append(data.meta_num_mask)
                        meta_cats.append(data.meta_cat)
                        meta_presents.append(data.meta_present)
                        # Past-only assessment age computed by clinical fold bridge.
                        # -1 means no eligible assessment for this feature set.
                        meta_age_days.append(getattr(data, "meta_age_days", torch.tensor([-1.0])))
                        meta_baseline_nums.append(data.meta_baseline_num)
                        meta_baseline_masks.append(data.meta_baseline_mask)
                        meta_months_since_baselines.append(data.meta_months_since_baseline)
                        current_visit_month = float(data.visit_months.view(-1)[0].item())
                        previous_gap = 0.0 if previous_visit_month is None else max(current_visit_month - previous_visit_month, 0.0)
                        meta_months_since_previous.append(torch.tensor([previous_gap], dtype=torch.float32))
                        previous_visit_month = current_visit_month
                    if self.use_ollama:
                        ollama_embeddings.append(data.ollama_emb)
                        ollama_presents.append(data.ollama_present)

                prev_dx_seq_raw.append(torch.tensor(dx_values, dtype=torch.long))
                if self.use_metadata:
                    meta_num_seq_raw.append(torch.stack(meta_nums).float())
                    meta_mask_seq_raw.append(torch.stack(meta_masks).float())
                    meta_cat_seq_raw.append(torch.stack(meta_cats).long())
                    meta_present_seq_raw.append(torch.stack(meta_presents).float())
                    meta_age_days_seq_raw.append(torch.stack(meta_age_days).float())
                    meta_baseline_num_seq_raw.append(torch.stack(meta_baseline_nums).float())
                    meta_baseline_mask_seq_raw.append(torch.stack(meta_baseline_masks).float())
                    meta_months_since_baseline_seq_raw.append(torch.stack(meta_months_since_baselines).float())
                    meta_months_since_previous_seq_raw.append(torch.stack(meta_months_since_previous).float())
                if self.use_ollama:
                    ollama_seq_raw.append(torch.stack(ollama_embeddings).float())
                    ollama_present_seq_raw.append(torch.stack(ollama_presents).float())

            if not all_data:
                continue

            big_batch = Batch.from_data_list(all_data).to(self.device)
            change_batch = change_valid = None
            if getattr(self.encoder, "use_graph_changes", False):
                from longitudinal_innovation import make_innovation_batch
                change_batch, change_valid = make_innovation_batch(
                    all_data, visit_to_window, self.encoder.mode,
                    floor_years=self.encoder.floor_years,
                    clip=self.encoder.clip,
                    node_math=self.encoder.node_math, edge_math=self.encoder.edge_math,
                )
                change_batch=change_batch.to(self.device)
                change_valid=change_valid.to(self.device)
            if self.use_time_features:
                normalized_gaps = normalize_time_gaps(
                    np.asarray(time_gaps, dtype=np.float32),
                    method=self.time_normalization,
                )
                per_graph_time = [normalized_gaps[i] for i in visit_to_window]
                time_features = torch.tensor(
                    per_graph_time, dtype=torch.float32, device=self.device
                )
            else:
                time_features = None

            _innovation_kwargs = (
                {"change_batch":change_batch, "change_valid":change_valid}
                if change_batch is not None else {}
            )
            all_embeddings = self.encoder(
                big_batch.x,
                big_batch.edge_index,
                big_batch.batch,
                time_features,
                edge_weight=big_batch.edge_weight,
                **_innovation_kwargs,
            )

            sequences = []
            for window_index in range(len(batch_windows)):
                embeddings = [
                    all_embeddings[i]
                    for i, assigned_window in enumerate(visit_to_window)
                    if assigned_window == window_index
                ]
                sequences.append(torch.stack(embeddings))
            graph_seq, lengths = self._pad_graph_sequences(sequences)

            batch = {
                "graph_seq": graph_seq,
                "lengths": lengths,
                "labels": torch.tensor(labels, dtype=torch.long, device=self.device),
                "subjects": subjects,
                "segments": segments,
                "target_dates": target_dates,
                "input_dates": input_dates,
                "time_gaps_raw": torch.tensor(
                    time_gaps, dtype=torch.float32, device=self.device
                ),
                "transitions": transitions,
                "prev_dx": prev_dx_labels,
                "target_dx": target_dx_labels,
            }

            if self.use_prev_dx:
                batch["prev_dx_seq"] = self._pad_visit_tensors(
                    prev_dx_seq_raw, pad_value=0, dtype=torch.long
                )
            if self.use_metadata:
                batch["meta_num_seq"] = self._pad_visit_tensors(
                    meta_num_seq_raw, dtype=torch.float32
                )
                batch["meta_mask_seq"] = self._pad_visit_tensors(
                    meta_mask_seq_raw, dtype=torch.float32
                )
                batch["meta_cat_seq"] = self._pad_visit_tensors(
                    meta_cat_seq_raw, dtype=torch.long
                )
                batch["meta_present_seq"] = self._pad_visit_tensors(
                    meta_present_seq_raw, dtype=torch.float32
                )
                batch["meta_age_days_seq"] = self._pad_visit_tensors(
                    meta_age_days_seq_raw, pad_value=-1, dtype=torch.float32
                )
                batch["meta_baseline_num_seq"] = self._pad_visit_tensors(
                    meta_baseline_num_seq_raw, dtype=torch.float32
                )
                batch["meta_baseline_mask_seq"] = self._pad_visit_tensors(
                    meta_baseline_mask_seq_raw, dtype=torch.float32
                )
                batch["meta_months_since_baseline_seq"] = self._pad_visit_tensors(
                    meta_months_since_baseline_seq_raw, dtype=torch.float32
                )
                batch["meta_months_since_previous_seq"] = self._pad_visit_tensors(
                    meta_months_since_previous_seq_raw, dtype=torch.float32
                )
            if self.use_ollama:
                batch["ollama_seq"] = self._pad_visit_tensors(
                    ollama_seq_raw, dtype=torch.float32
                )
                batch["ollama_present_seq"] = self._pad_visit_tensors(
                    ollama_present_seq_raw, dtype=torch.float32
                )
            yield batch
