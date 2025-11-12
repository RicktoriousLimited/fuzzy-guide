"""NSCTX model assembly."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from .layers.embedding import ModalityPooler, ModalityTextEncoder, SharedProjection
from .layers.fusion import ContrastiveAlignmentLoss, ModalityFusion
from .layers.graph import GraphReasoner, SemanticGraphBuilder
from .layers.meta import MetaLearner
from .layers.reasoning import ElasticWeightConsolidation, NeuroSymbolicReasoner
from .layers.transformer import TransformerStack


@dataclass
class NSCTXConfig:
    vocab_size: int
    d_model: int = 64
    n_heads: int = 4
    dim_feedforward: int = 128
    n_roles: int = 8
    n_relations: int = 3
    n_outputs: int = 3
    fusion_dim: int = 64
    transformer_layers: int = 2


class NSCTXModel(nn.Module):
    def __init__(self, config: NSCTXConfig):
        super().__init__()
        self.config = config
        self.text_encoder = ModalityTextEncoder(config.vocab_size, config.d_model)
        self.shared_proj = SharedProjection(config.d_model, config.fusion_dim)
        self.pool = ModalityPooler()
        self.fusion = ModalityFusion(config.fusion_dim, config.fusion_dim, ("text",))
        self.context_stack = TransformerStack(
            depth=config.transformer_layers,
            d_model=config.fusion_dim,
            n_heads=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            n_roles=config.n_roles,
        )
        self.graph_builder = SemanticGraphBuilder(config.fusion_dim, config.n_relations)
        self.graph_reasoner = GraphReasoner(config.fusion_dim, config.n_relations)
        self.reasoner = NeuroSymbolicReasoner(config.fusion_dim, config.n_outputs)
        self.meta = MetaLearner(self.reasoner)
        self.ewc = ElasticWeightConsolidation()
        self.contrastive_loss = ContrastiveAlignmentLoss()

    def encode_text(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.text_encoder(tokens)
        pooled = self.pool(emb, mask)
        projected = self.shared_proj(pooled)
        return projected, emb

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        text_tokens = batch["text"]
        mask = batch.get("mask")
        mask_tensor = mask if mask is not None else torch.ones_like(text_tokens, dtype=torch.float)
        text_repr, text_context = self.encode_text(text_tokens, mask_tensor.float())
        fused, fusion_weights = self.fusion({"text": text_repr})
        roles = torch.zeros(text_tokens.size(0), text_tokens.size(1), dtype=torch.long, device=text_tokens.device)
        context_input = text_context + fused.unsqueeze(1)
        contextual = self.context_stack(context_input, roles)
        nodes, edges = self.graph_builder(text_tokens, contextual)
        hub = self.graph_reasoner(nodes, edges)
        logits = self.reasoner(hub)
        return {
            "logits": logits,
            "fusion_weights": fusion_weights,
            "hub": hub,
        }

    def compute_losses(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = outputs["logits"]
        hub = outputs["hub"]
        losses: Dict[str, torch.Tensor] = {}
        losses["lm"] = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        losses["rule"] = self.reasoner.rule_loss(logits)
        losses["ewc"] = self.ewc.penalty(self.reasoner)
        losses["consistency"] = torch.mean(torch.abs(hub - hub.detach()))
        losses["total"] = (
            losses["lm"]
            + 0.1 * losses["rule"]
            + 0.01 * losses["ewc"]
            + 0.01 * losses["consistency"]
        )
        return losses

    def adapt(self, loss: torch.Tensor) -> None:
        self.meta(loss)
