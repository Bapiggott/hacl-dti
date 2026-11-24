from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GraphNorm,
    MessagePassing,
    SAGEConv,
    TopKPooling,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import scatter


class WeightedSAGEConv(MessagePassing):
    """GraphSAGE-style convolution with a normalized weighted-neighbor mean.

    For node ``i``::

        m_i = sum_j w[j,i] * x_j / max(sum_j w[j,i], eps)
        out_i = W_neigh(m_i) + W_root(x_i)

    Setting every edge weight to one recovers ordinary mean-neighbor GraphSAGE
    behavior, while preserving a separate transform for the root node.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        normalize: bool = False,
        root_weight: bool = True,
        bias: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(aggr="add")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.normalize = bool(normalize)
        self.root_weight = bool(root_weight)
        self.eps = float(eps)

        self.lin_neigh = nn.Linear(self.in_channels, self.out_channels, bias=bias)
        self.lin_root: Optional[nn.Linear]
        if self.root_weight:
            self.lin_root = nn.Linear(self.in_channels, self.out_channels, bias=False)
        else:
            self.lin_root = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin_neigh.reset_parameters()
        if self.lin_root is not None:
            self.lin_root.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected x=[N,F], got {tuple(x.shape)}")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"Expected edge_index=[2,E], got {tuple(edge_index.shape)}")

        num_edges = int(edge_index.size(1))
        if edge_weight is None:
            edge_weight = x.new_ones(num_edges)
        else:
            edge_weight = edge_weight.to(device=x.device, dtype=x.dtype).reshape(-1)
            if edge_weight.numel() != num_edges:
                raise ValueError(
                    f"edge_weight contains {edge_weight.numel()} values for {num_edges} edges"
                )
            if not torch.isfinite(edge_weight).all():
                raise ValueError("edge_weight contains NaN or infinity")
            edge_weight = edge_weight.clamp_min(0.0)

        neighbor_sum = self.propagate(
            edge_index,
            x=x,
            edge_weight=edge_weight,
            size=(x.size(0), x.size(0)),
        )
        destination = edge_index[1]
        weight_sum = scatter(
            edge_weight,
            destination,
            dim=0,
            dim_size=x.size(0),
            reduce="sum",
        ).clamp_min(self.eps)
        neighbor_mean = neighbor_sum / weight_sum.unsqueeze(-1)

        out = self.lin_neigh(neighbor_mean)
        if self.lin_root is not None:
            out = out + self.lin_root(x)
        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)
        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return x_j * edge_weight.unsqueeze(-1)


class GraphNeuralNetwork(nn.Module):
    """AAL graph encoder with optional edge weights and optional TopK pooling.

    Supported layers:
      * ``GraphSAGE``: canonical PyG SAGEConv; topology only.
      * ``WeightedGraphSAGE``: weighted-neighbor GraphSAGE-style convolution.
      * ``GCN``: PyG GCNConv; uses edge weights.
      * ``GAT``: PyG GATConv; topology only in this implementation.

    The graph readout is always global mean concatenated with global max, so
    ``output_graph_dim == 2 * output_dim`` regardless of TopK usage.
    """

    def __init__(
        self,
        input_dim: int = 100,
        hidden_dim: int = 128,
        output_dim: int = 256,
        dropout: float = 0.5,
        use_topk_pooling: bool = True,
        topk_ratio: float = 0.5,
        layer_type: str = "GCN",
        num_layers: int = 3,
        activation: str = "relu",
        use_time_features: bool = False,
        dropedge_p: float = 0.0,
    ) -> None:
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("num_layers must be at least 1")

        self.use_topk_pooling = bool(use_topk_pooling)
        self.topk_ratio = float(topk_ratio)
        self.dropedge_p = float(dropedge_p)
        self.num_layers = int(num_layers)
        self.layer_type = str(layer_type)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.output_graph_dim = self.output_dim * 2
        self.use_time_features = bool(use_time_features)

        activation_map = {
            "relu": F.relu,
            "leaky_relu": F.leaky_relu,
            "elu": F.elu,
            "gelu": F.gelu,
        }
        if activation not in activation_map:
            raise ValueError(f"Unknown activation: {activation}")
        self.activation_fn = activation_map[activation]

        self.convs = nn.ModuleList()
        self.gns = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == self.num_layers - 1 else hidden_dim

            if self.layer_type == "GCN":
                conv = GCNConv(in_dim, out_dim)
            elif self.layer_type == "GAT":
                conv = GATConv(in_dim, out_dim)
            elif self.layer_type == "GraphSAGE":
                conv = SAGEConv(in_dim, out_dim)
            elif self.layer_type == "WeightedGraphSAGE":
                conv = WeightedSAGEConv(in_dim, out_dim)
            else:
                raise ValueError(f"Unknown layer type: {self.layer_type}")

            self.convs.append(conv)
            self.gns.append(GraphNorm(out_dim))

        self.dropout = nn.Dropout(p=float(dropout))

        self.topk_pools = nn.ModuleList()
        if self.use_topk_pooling:
            safe_ratio = max(0.3, min(1.0, self.topk_ratio))
            for i in range(self.num_layers):
                pool_dim = output_dim if i == self.num_layers - 1 else hidden_dim
                self.topk_pools.append(
                    TopKPooling(pool_dim, ratio=safe_ratio, min_score=None)
                )
            print(
                f"TopK pooling initialized with {self.num_layers} layers, "
                f"safe ratio={safe_ratio:.3f}"
            )

        self.pool_mean = global_mean_pool
        self.pool_max = global_max_pool

        if self.use_time_features:
            time_dim = 32
            self.time_projection = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(16, time_dim),
            )
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.output_graph_dim + time_dim, self.output_graph_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
            )
        else:
            self.time_projection = None
            self.fusion_layer = None

    @property
    def uses_edge_weight(self) -> bool:
        return self.layer_type in {"GCN", "WeightedGraphSAGE"}

    def _drop_edges(
        self,
        edge_index: Tensor,
        edge_weight: Optional[Tensor],
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Drop matching edge indices and edge weights during training."""
        if (
            not self.training
            or self.dropedge_p <= 0.0
            or edge_index.numel() == 0
        ):
            return edge_index, edge_weight

        keep = torch.rand(edge_index.size(1), device=edge_index.device) >= self.dropedge_p
        if int(keep.sum().item()) == 0:
            return edge_index, edge_weight
        edge_index = edge_index[:, keep]
        if edge_weight is not None:
            edge_weight = edge_weight[keep]
        return edge_index, edge_weight

    def _apply_conv(
        self,
        conv: nn.Module,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor],
    ) -> Tensor:
        if self.layer_type == "GCN":
            return conv(x, edge_index, edge_weight=edge_weight)
        if self.layer_type == "WeightedGraphSAGE":
            return conv(x, edge_index, edge_weight=edge_weight)
        return conv(x, edge_index)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        time_to_predict: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        edge_weight = (
            edge_weight.to(device=x.device, dtype=x.dtype).reshape(-1)
            if edge_weight is not None
            else None
        )
        edge_index, edge_weight = self._drop_edges(edge_index, edge_weight)

        for i in range(self.num_layers):
            x = self._apply_conv(self.convs[i], x, edge_index, edge_weight)
            x = self.gns[i](x, batch)
            x = self.activation_fn(x)
            x = self.dropout(x)

            if self.use_topk_pooling:
                x, edge_index, edge_weight, batch, _perm, _score = self.topk_pools[i](
                    x,
                    edge_index,
                    edge_attr=edge_weight,
                    batch=batch,
                )

        graph_embedding = torch.cat(
            [self.pool_mean(x, batch), self.pool_max(x, batch)], dim=1
        )

        if self.use_time_features and time_to_predict is not None:
            if time_to_predict.dim() == 1:
                time_to_predict = time_to_predict.unsqueeze(1)
            time_embedding = self.time_projection(time_to_predict.float())
            graph_embedding = self.fusion_layer(
                torch.cat([graph_embedding, time_embedding], dim=1)
            )
        return graph_embedding

    def load_state_dict_flexible(self, state_dict, strict: bool = False):
        """Load matching tensors and skip architecture-specific mismatches."""
        current = self.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in current and tuple(value.shape) == tuple(current[key].shape)
        }
        skipped = sorted(set(state_dict) - set(compatible))
        missing = sorted(set(current) - set(compatible))
        if skipped:
            print(f"[load_state_dict_flexible] skipped {len(skipped)} source tensors")
        if missing:
            print(f"[load_state_dict_flexible] initialized {len(missing)} new tensors")
        return super().load_state_dict(compatible, strict=strict)
