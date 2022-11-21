"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch.nn as nn
from dgl.nn.pytorch import GATConv


class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, False, self.activation))

    def forward(self, inputs):
        heads = []
        h = inputs
        # get hidden_representation
        for l in range(self.num_layers):
            temp = h.flatten(1)
            h =self.gat_layers[l](self.g, temp)
        # get heads
        for i in range(h.shape[1]):
            heads.append(h[:, i])
        return heads
