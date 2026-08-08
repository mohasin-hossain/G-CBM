"""MLP classifier on the K-dim concept-bottleneck vector z.

Expects one-node graphs from cbm_graph (feat shape [1, K]).
"""

import torch
import torch.nn as nn
import dgl


class ConceptBottleneckMLP(nn.Module):
    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        num_heads: int = None,
        out_dim: int = 2,
        feat_drop: float = 0.0,
        node_drop: float = 0.0,
    ):
        super().__init__()
        self.feat_drop = feat_drop
        self.node_drop = node_drop
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.ELU(),
            nn.Linear(out_feats, out_feats),
            nn.ELU(),
        )
        self.classify = nn.Linear(out_feats, out_dim)

    def forward(self, graph, nfeats):
        with graph.local_scope():
            graph.ndata["h"] = nfeats
            hg = dgl.mean_nodes(graph, "h")
        h = self.mlp(hg)
        logits = self.classify(h)
        attn_placeholder = torch.zeros(logits.shape[0], device=logits.device)
        return logits, attn_placeholder, h
