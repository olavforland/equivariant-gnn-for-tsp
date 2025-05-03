import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class CircularHarmonicConv(nn.Module):
    """Configurable CircularHarmonicsConvGNN
    """
    
    def __init__(self, hidden_dim, M=3, N_b=8):
        super(CircularHarmonicConv, self).__init__()

        self.M = M
        self.N_b = N_b # num bessel roots

        # learnable, ordered Bessel roots
        self.g = nn.Parameter(torch.arange(1, N_b+1, dtype=torch.float32))

        self.self_interaction = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.radial_mlp = nn.Sequential(
            nn.Linear(self.N_b, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, M+1),
        )

    def forward(self, pos):
        """
        Args:
            pos: Input node positions (B x V x 2)
        Returns: 
            Edge convolutions (B x V x V x H)
        """
        B, V, _ = pos.shape
        delta = pos.unsqueeze(2) - pos.unsqueeze(1)  # B x V x V x 2
        r = delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # B x V x V x 1
        rc = r.max().detach() # scalar cutoff (max distance)

        # Encode each edge with Bessel functions
        # This encapsulates the radial part of the edge features
        g_pos = F.softplus(self.g)
        b = torch.cumsum(g_pos, dim=0).view(1, 1, 1, -1)
        bessel = 2 * torch.sin(b * torch.pi * r / rc) / (rc * r)
        R = self.radial_mlp(bessel)  

        # Encode the angular part of the edge features
        theta = torch.atan2(delta[..., 1:2], delta[..., 0:1] + 1e-6)  # B x V x V x 1
        m = torch.arange(0, self.M+1, device=pos.device) \
                    .view(1,1,1,-1) # (1,1,1,M+1)
        cosm = torch.cos(m * theta) # (B,V,V,M+1)
        sinm = torch.sin(m * theta) # (B,V,V,M+1)

        # Combine radial and angular part in the final edge‐feature
        #    – first channel is R[...,0]*1
        #    – for m>=1 use R[...,m]*cos(mθ) and R[...,m]*sin(mθ)
        cos_feat = R * cosm # (B,V,V,M+1)
        sin_feat = R[...,1:] * sinm[...,1:] # (B,V,V,M)

        edge_features = torch.cat([cos_feat, sin_feat], dim=-1) # (B,V,V,2*M+1)
        return edge_features


    
class InteractionBlock(nn.Module): 
    """Based on Nequip: self-interaction → conv → concat → self-interaction."""

    def __init__(self, hidden_dim, M, aggregation="max"):
        super(InteractionBlock, self).__init__()
        self.aggregation = aggregation
        self.hidden_dim = hidden_dim
        self.pre_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.SiLU(),
        )
        self.post_fc = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), 
            nn.SiLU(),
        )
        self.skip_interact = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.SiLU(),
        )
        self.non_linearity = nn.SiLU()
        # self.order_weights = nn.Parameter(torch.ones(1, 1, 1, 2*M+1))


    def forward(self, h, e, graph):
        """
        h:    (B, V, H) node features
        e:    (B,V,V,H)  edge features
        graph:(B, V, V)  0=edge,1=no edge
        """
        # self interact
        h0 = self.pre_fc(h) # (B,V,H)

        # convolution
        # m = torch.einsum("bvum,buh->bvuh", e * self.order_weights, h0) # (B,V,V,H)
        m = e * h0.unsqueeze(1) # (B,V,V,H)

        m = m.masked_fill(graph.unsqueeze(-1).bool(), 0)  
        if self.aggregation=="mean":
            deg = (1-graph).sum(-1,keepdim=True)
            neigh = m.sum(2)/deg
        elif self.aggregation=="max":
            neigh = m.max(2)[0]
        else:  # sum
            neigh = m.sum(2) 

        # concat & post self-interaction + residual
        h_cat = torch.cat([h0, neigh], dim=-1)            # (B,V,2H)
        h1 = self.post_fc(h_cat)                          # (B,V,H)
        return self.non_linearity(self.skip_interact(h) + h1)                                      # residual




# Refactored InteractionBlock ─────────────────────────

class CircularHarmonicsGNNEncoder(nn.Module):
    """Embed → L x InteractionBlock → OutputBlock (not shown)."""
    def __init__(self, n_layers, hidden_dim, aggregation="sum", norm="layer", 
                 learn_norm=True, track_norm=False, gated=True, *args, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim    
        M = 5
        self.conv = CircularHarmonicConv(hidden_dim, M=M, N_b=8)
        # stack InteractionBlocks
        self.blocks = nn.ModuleList([
            InteractionBlock(hidden_dim, M=M, aggregation='sum')
            for _ in range(n_layers)
        ])
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.SiLU(),
        )
        self.proj = nn.Linear(2*M+1, hidden_dim)

    def forward(self, pos, graph):
        e = self.conv(pos)         # (B,V,V,2M+1)
        e = self.proj(e)         # (B,V,V,H)
        h = e.sum(1) # torch.zeros(size=pos.shape[:-1] + (self.hidden_dim, )) # (B,V,H)
        for block in self.blocks:
            h = block(h, e, graph) # note: pass pos=x to conv
        return self.output(h)