
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool

class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, dim: int): super().__init__(); self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device; half = self.dim // 2
        if half == 0: return torch.zeros(t.shape[0], 0, device=device)
        fac = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / (half - 1 + 1e-8)))
        ang = t.float().unsqueeze(-1) * fac.unsqueeze(0)
        pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if self.dim % 2 == 1: pe = torch.cat([pe, torch.zeros(pe.shape[0],1,device=device)], dim=-1)
        return pe

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x): return self.net(x)

class SBlock(nn.Module):
    def __init__(self, dim, heads=4, edge_dim_s=0, dropout=0.1):
        super().__init__()
        self.s_attn = TransformerConv(dim, dim//heads, heads=heads, edge_dim=(edge_dim_s if edge_dim_s>0 else None), beta=True, dropout=dropout)
        self.norm_s = nn.LayerNorm(dim); self.ff = MLP(dim, 4*dim, dim, dropout=dropout); self.norm_ff = nn.LayerNorm(dim)
    def forward(self, x, edge_index_s, edge_attr_s):
        xs = self.s_attn(x, edge_index_s, edge_attr=edge_attr_s) if (edge_index_s is not None and edge_index_s.numel()>0) else torch.zeros_like(x)
        x = self.norm_s(x + xs); x = self.norm_ff(x + self.ff(x)); return x

class TactileGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_layers=4, heads=4, edge_dim_s=0, time_emb_dim=32, dropout=0.1, readout="meanmax"):
        super().__init__()
        self.time_enc = SinusoidalTimeEncoding(time_emb_dim) if time_emb_dim>0 else None
        self.input_proj = nn.Linear(in_dim + (time_emb_dim if self.time_enc is not None else 0), hidden_dim)
        self.blocks = nn.ModuleList([SBlock(hidden_dim, heads=heads, edge_dim_s=edge_dim_s, dropout=dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout); self.readout = readout
        out_in = hidden_dim * (2 if readout=="meanmax" else 1)
        self.head = MLP(out_in, hidden_dim, 2, dropout=dropout)
    def forward(self, data):
        x = data.x
        if hasattr(data,'t') and self.time_enc is not None:
            x = torch.cat([x, self.time_enc(data.t)], dim=-1)
        x = self.input_proj(x); x = F.relu(x); x = self.dropout(x)
        e_s = getattr(data,'edge_index_s', None); ea_s = getattr(data,'edge_attr_s', None)
        for blk in self.blocks: x = blk(x, e_s, ea_s)
        # if hasattr(data,'t'):
        #     mx = data.t.max(); m = (data.t==mx); x_last = x[m]; b_last = data.batch[m]
        # else:
        #     x_last = x; b_last = data.batch
        if hasattr(data, 't'):
            # t 逐图最大值（先转 float 再做图级 max）
            t_float = data.t.to(x.dtype).unsqueeze(-1)           # [N,1]
            tmax_per_g = global_max_pool(t_float, data.batch).squeeze(-1)   # [G]
            m = (t_float.squeeze(-1) == tmax_per_g[data.batch])  # [N]
            x_last = x[m]; b_last = data.batch[m]
        else:
            x_last = x; b_last = data.batch
        if self.readout=="mean": g = global_mean_pool(x_last, b_last)
        elif self.readout=="max": g = global_max_pool(x_last, b_last)
        else: g = torch.cat([global_mean_pool(x_last, b_last), global_max_pool(x_last, b_last)], dim=-1)
        return self.head(g)
