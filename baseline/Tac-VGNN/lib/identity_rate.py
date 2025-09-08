# # identity_rate.py
# import numpy as np, torch

# def _split_frames(d, N):
#     T = d.x.size(0)//N
#     XY = d.x[:, :2].view(T, N, 2).detach().cpu().numpy()
#     return XY, T

# # def _temporal_edges_between(d, t, N):
# #     i0, j0 = t*N, (t+1)*N
# #     if hasattr(d, "temporal_edge_index") and d.temporal_edge_index is not None:
# #         tei = d.temporal_edge_index
# #     else:
# #         ei = d.edge_index
# #         tei = ei[:, (ei[0]-ei[1]).abs()==N]
# #     if tei.numel()==0: return np.empty(0, np.int64), np.empty(0, np.int64)
# #     m = (tei[0]>=i0)&(tei[0]<i0+N)&(tei[1]>=j0)&(tei[1]<j0+N)
# #     te = tei[:, m].detach().cpu().numpy()
# #     if te.shape[1]==0: return np.empty(0, np.int64), np.empty(0, np.int64)
# #     return (te[0]-i0).astype(np.int64), (te[1]-j0).astype(np.int64)

# def identity_rate(d, N=331, mode="geom_over_edges", use_hungarian=True):
#     """
#     mode:
#       - 'edge_index'       -> E-ICR（边索引自洽率，通常=1）
#       - 'nn_over_frames'   -> ICR-NN（几何最近邻，一般 0.8~0.95）
#       - 'geom_over_edges'  -> GIR-E（推荐，默认）
#     """
#     XY, T = _split_frames(d, N)
#     if T<=1: return 0.0

#     if mode == "edge_index":
#         # 只看你连的边是否 i->i
#         if hasattr(d, "temporal_edge_index") and d.temporal_edge_index is not None:
#             te = d.temporal_edge_index
#         else:
#             ei = d.edge_index
#             te = ei[:, (ei[0]-ei[1]).abs()==N]
#         if te.numel()==0: return 0.0
#         src = (te[0] % N); dst = (te[1] % N)
#         return float((src==dst).float().mean().item())

#     if mode == "nn_over_frames":
#         vals=[]
#         for t in range(T-1):
#             A = torch.tensor(XY[t]); B = torch.tensor(XY[t




# identity_rate.py
import numpy as np, torch

def _split_frames(d, N):
    T = int(d.x.size(0) // N)
    XY = d.x[:, :2].view(T, N, 2).detach().cpu().numpy()
    return XY, T

def _temporal_edges_between(d, t, N):
    """取 t→t+1 的时序边 (src@t, dst@t+1)，索引映射到 [0,N)"""
    i0, j0 = t*N, (t+1)*N
    # 优先用单独保存的 temporal_edge_index
    if hasattr(d, "temporal_edge_index") and (getattr(d, "temporal_edge_index") is not None):
        tei = d.temporal_edge_index
    else:
        ei = d.edge_index
        tei = ei[:, (ei[0]-ei[1]).abs() == N]
    if tei.numel() == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    tei = tei.detach().cpu().numpy()
    m = (tei[0] >= i0) & (tei[0] < i0+N) & (tei[1] >= j0) & (tei[1] < j0+N)
    te = tei[:, m]
    if te.shape[1] == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return (te[0]-i0).astype(np.int64), (te[1]-j0).astype(np.int64)

def kept_frac(d, N):
    """实际保留的时序边 / 理论最大 (T-1)*N"""
    XY, T = _split_frames(d, N)
    if T <= 1: return 0.0
    if hasattr(d, "temporal_edge_index") and (getattr(d,"temporal_edge_index") is not None):
        te_cnt = int(d.temporal_edge_index.size(1))
    else:
        ei = d.edge_index
        te_cnt = int(((ei[0]-ei[1]).abs()==N).sum().item())
    return te_cnt / float((T-1)*N)

def identity_rate(d, N=331, mode="geom_over_edges", use_hungarian=True):
    """
    统一的 IdentityRate 入口：
      - mode='edge_index'       -> E-ICR：只看 i→i 的索引自洽率（常=1.0）
      - mode='nn_over_frames'   -> ICR-NN：基于几何最近邻，不看边
      - mode='geom_over_edges'  -> GIR-E：几何匹配后，仅在“保留的时序边”集合内评估（推荐）
    """
    XY, T = _split_frames(d, N)
    if T <= 1: return 0.0

    if mode == "edge_index":
        if hasattr(d, "temporal_edge_index") and (getattr(d,"temporal_edge_index") is not None):
            te = d.temporal_edge_index
        else:
            ei = d.edge_index
            te = ei[:, (ei[0]-ei[1]).abs()==N]
        if te.numel()==0: return 0.0
        src = (te[0] % N); dst = (te[1] % N)
        return float((src == dst).float().mean().item())

    if mode == "nn_over_frames":
        vals = []
        for t in range(T-1):
            A = XY[t]      # (N,2)
            B = XY[t+1]    # (N,2)
            D = ((A[:,None,:]-B[None,:,:])**2).sum(-1)**0.5  # (N,N)
            nn = D.argmin(axis=1)                            # (N,)
            vals.append(float((nn == np.arange(N)).mean()))
        return float(np.mean(vals)) if len(vals)>0 else 0.0

    if mode == "geom_over_edges":
        # 先准备几何匹配
        if use_hungarian:
            try:
                from scipy.spatial.distance import cdist
                from scipy.optimize import linear_sum_assignment
                HAS = True
            except Exception:
                HAS = False
                use_hungarian = False
        else:
            HAS = False

        ok, tot = 0, 0
        for t in range(T-1):
            src_idx, dst_idx = _temporal_edges_between(d, t, N)
            if src_idx.size == 0: 
                continue
            A = XY[t]; B = XY[t+1]
            if use_hungarian and HAS:
                C = cdist(A, B)                     # (N,N)
                _, col = linear_sum_assignment(C)   # col[i] 是 i@t 对应到 t+1 的 j
                mapping = np.asarray(col, dtype=np.int64)
            else:
                D = ((A[:,None,:]-B[None,:,:])**2).sum(-1)
                mapping = D.argmin(axis=1).astype(np.int64)
            ok  += int((mapping[src_idx] == dst_idx).sum())
            tot += int(src_idx.size)
        return ok / max(1, tot)

    raise ValueError(f"Unknown mode={mode}")

def dataset_summary(ds, N=331, name="set", use_hungarian=True):
    """打印三种口径 + kept_frac 的数据集汇总，口径统一、便于贴给老师"""
    gir = []   # 几何按边（推荐）
    nnr = []   # 最近邻
    eic = []   # 边索引自洽
    kfs = []
    for d in ds:
        gir.append(identity_rate(d, N, mode="geom_over_edges", use_hungarian=use_hungarian))
        nnr.append(identity_rate(d, N, mode="nn_over_frames"))
        eic.append(identity_rate(d, N, mode="edge_index"))
        kfs.append(kept_frac(d, N))
    if len(ds)==0:
        print(f"[{name}] empty"); return
    print(f"[{name}] GIR-E (geom_over_edges) mean/min/max = {np.mean(gir):.3f} / {np.min(gir):.3f} / {np.max(gir):.3f}")
    print(f"[{name}] ICR-NN (nn_over_frames) mean/min/max = {np.mean(nnr):.3f} / {np.min(nnr):.3f} / {np.max(nnr):.3f}")
    print(f"[{name}] E-ICR (edge_index)       mean/min/max = {np.mean(eic):.3f} / {np.min(eic):.3f} / {np.max(eic):.3f}")
    print(f"[{name}] kept_frac(mean) = {np.mean(kfs):.3f}")
