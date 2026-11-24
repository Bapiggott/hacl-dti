#!/usr/bin/env python3
"""Advanced graph encoder options for DTI-STGNN.

This file is optional. The original model.py GraphNeuralNetwork is still supported.
The advanced encoder adds residual connections, optional Jumping Knowledge-style
concatenation, and additional PyG layer types when available.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv,
    GINConv,
    GraphNorm,
    global_mean_pool,
    global_max_pool,
)

try:
    from torch_geometric.nn import GATv2Conv
except Exception:
    GATv2Conv = None

try:
    from torch_geometric.nn import TransformerConv
except Exception:
    TransformerConv = None


class AdvancedGraphNeuralNetwork(nn.Module):
    """A slightly stronger graph encoder for AAL DTI graphs.

    Output is a graph-level embedding made from global mean + global max pooling.
    If jk='last', output dimension is output_dim * 2.
    If jk='concat', output dimension is sum(layer_dims) * 2.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.20,
        layer_type: str = "GATv2",
        num_layers: int = 3,
        activation: str = "gelu",
        residual: bool = True,
        jk: str = "last",
        heads: int = 4,
        use_time_features: bool = False,
    ):
        super().__init__()
        self.layer_type = layer_type
        self.num_layers = int(num_layers)
        self.dropout = nn.Dropout(dropout)
        self.residual = bool(residual)
        self.jk = jk
        self.use_time_features = bool(use_time_features)

        act = activation.lower()
        self.activation_fn = {
            "relu": F.relu,
            "leaky_relu": F.leaky_relu,
            "elu": F.elu,
            "gelu": F.gelu,
        }.get(act, F.gelu)

        dims = []
        for i in range(self.num_layers):
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = hidden_dim
            out_dim = output_dim if i == self.num_layers - 1 else hidden_dim
            dims.append((in_dim, out_dim))

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_projs = nn.ModuleList()

        for in_dim, out_dim in dims:
            conv, actual_out = self._make_conv(layer_type, in_dim, out_dim, heads=heads)
            self.convs.append(conv)
            self.norms.append(GraphNorm(actual_out))
            if self.residual and in_dim != actual_out:
                self.res_projs.append(nn.Linear(in_dim, actual_out))
            else:
                self.res_projs.append(nn.Identity())

        self.layer_out_dims = [n.num_features if hasattr(n, "num_features") else n.normalized_shape[0] if hasattr(n, "normalized_shape") else dims[i][1] for i, n in enumerate(self.norms)]
        if self.jk == "concat":
            node_dim = sum(self.layer_out_dims)
        else:
            node_dim = self.layer_out_dims[-1]

        self.output_graph_dim = node_dim * 2

        if self.use_time_features:
            time_dim = 32
            self.time_projection = nn.Sequential(
                nn.Linear(1, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, time_dim)
            )
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.output_graph_dim + time_dim, self.output_graph_dim),
                nn.ReLU(), nn.Dropout(dropout)
            )

    def _make_conv(self, layer_type: str, in_dim: int, out_dim: int, heads: int):
        lt = layer_type.lower()
        if lt == "gcn":
            return GCNConv(in_dim, out_dim), out_dim
        if lt in {"graphsage", "sage"}:
            return SAGEConv(in_dim, out_dim), out_dim
        if lt == "gin":
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )
            return GINConv(mlp), out_dim
        if lt == "gat":
            # concat=False keeps the output dimension exactly out_dim.
            return GATConv(in_dim, out_dim, heads=heads, concat=False, dropout=0.1), out_dim
        if lt == "gatv2":
            if GATv2Conv is None:
                raise ImportError("GATv2Conv is not available in your torch_geometric install.")
            return GATv2Conv(in_dim, out_dim, heads=heads, concat=False, dropout=0.1), out_dim
        if lt == "transformer":
            if TransformerConv is None:
                raise ImportError("TransformerConv is not available in your torch_geometric install.")
            return TransformerConv(in_dim, out_dim, heads=heads, concat=False, dropout=0.1), out_dim
        raise ValueError(f"Unknown advanced layer type: {layer_type}")

    def forward(self, x, edge_index, batch, time_to_predict=None):
        outs = []
        for conv, norm, res_proj in zip(self.convs, self.norms, self.res_projs):
            x_in = x
            x = conv(x, edge_index)
            x = norm(x, batch)
            if self.residual:
                x = x + res_proj(x_in)
            x = self.activation_fn(x)
            x = self.dropout(x)
            outs.append(x)

        if self.jk == "concat":
            x_graph = torch.cat(outs, dim=-1)
        else:
            x_graph = outs[-1]

        x_mean = global_mean_pool(x_graph, batch)
        x_max = global_max_pool(x_graph, batch)
        graph_embedding = torch.cat([x_mean, x_max], dim=-1)

        if self.use_time_features and time_to_predict is not None:
            if len(time_to_predict.shape) == 1:
                time_to_predict = time_to_predict.unsqueeze(1)
            t = self.time_projection(time_to_predict)
            graph_embedding = self.fusion_layer(torch.cat([graph_embedding, t], dim=-1))

        return graph_embedding
