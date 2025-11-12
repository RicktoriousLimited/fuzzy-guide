"""Semantic graph induction and reasoning layers."""
from __future__ import annotations

import torch
import torch.nn as nn


class SemanticGraphBuilder(nn.Module):
    """Build a meaning graph from contextual embeddings."""

    def __init__(self, d_model: int, n_relations: int):
        super().__init__()
        self.node_proj = nn.Linear(d_model, d_model)
        self.edge_params = nn.Parameter(torch.randn(n_relations, d_model, d_model))
        self.n_relations = n_relations

    def forward(self, tokens: torch.Tensor, contextual: torch.Tensor):
        node_embeddings = torch.tanh(self.node_proj(contextual))
        edges = torch.einsum("bnd,rdk,bmk->brnm", node_embeddings, self.edge_params, node_embeddings)
        edges = torch.sigmoid(edges)
        return node_embeddings, edges


class GraphReasoner(nn.Module):
    """Graph neural network that performs message passing."""

    def __init__(self, d_model: int, n_relations: int, steps: int = 2):
        super().__init__()
        self.steps = steps
        self.n_relations = n_relations
        self.msg_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(n_relations)]
        )
        self.update = nn.GRUCell(d_model, d_model)

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        b, n, d = nodes.shape
        h = nodes.reshape(b * n, d)
        for _ in range(self.steps):
            agg = torch.zeros_like(h)
            nodes_reshaped = h.view(b, n, d)
            for rel, layer in enumerate(self.msg_layers):
                weights = edges[:, rel]
                agg_rel = torch.bmm(weights, nodes_reshaped)
                agg += layer(agg_rel.view(b * n, d))
            h = self.update(agg, h)
        return h.view(b, n, d).mean(dim=1)
