
# -*- coding: utf-8 -*-
"""
GCN-1f-S (frameify-all-frames) baseline builder
------------------------------------------------
Use this to align your single-frame baseline with a per-frame, spatial-only dataset
(the same assumption as your teacher's Tac-VGNN 1f-S).

WHAT YOU GET
------------
- frameify_all_frames(): expand sequences to per-frame PyG Data samples
- build_1f_S_datasets(): plug-and-play entry to build train/val datasets for VARIANT="1f_frameify"
- delaunay_edge_index(): robust Delaunay-to-edge_index (with safe fallbacks)
- assert_spatial_only(): sanity check to prevent accidental temporal leakage
- compute_mae_phys(): MAE in physical units after inverse transform

YOU MUST WIRE (TWO TINY HOOKS)
-------------------------------
1) get_num_frames(seq_item) -> int
2) extractor(seq_item, t) -> (pos_xy: np.ndarray[N,2], x_node: Optional[np.ndarray[N,F]], y_label: np.ndarray[D])
   - pos_xy should be in the SAME coordinate system you used for your teacher baseline (e.g., mm or normalized pixels)
   - x_node can be None; if None, we will use pos as features by default (you can change that)
   - y_label is the per-frame target (pose_2 / pose_6) **IN TRAINING SCALE**; MAE uses inverse_transform you pass in.

USAGE
-----
from frameify_1f_S_baseline import build_1f_S_datasets

train_dataset_run, val_dataset_run = build_1f_S_datasets(
    train_dataset_raw, val_dataset_raw,
    get_num_frames=my_get_num_frames,
    extractor=my_extractor,
    tip_num=None,              # keep None to match teacher
    add_self_loops=True,
    make_undirected=True,
    require_delaunay=True,     # set False to auto-fallback if scipy not installed
)

Then train your GCN on these per-frame samples. For evaluation:
pred_phys, gt_phys = inverse_transform_y(pred, scaler_y), inverse_transform_y(gt, scaler_y)
mae = compute_mae_phys(pred_phys, gt_phys)  # returns tensor with per-dimension MAE

Notes
-----
- This constructs PURE spatial graphs (S). NO temporal edges or features.
- Disable any node filtering like TIP_NUM while aligning with your teacher baseline.
"""

from typing import Callable, Iterable, List, Tuple, Optional, Any
import numpy as np
import warnings

try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.utils import add_self_loops
except Exception as e:
    raise ImportError("This module requires PyTorch and PyTorch Geometric. Please install them first.") from e

# -----------------------------
# Delaunay -> edge_index helper
# -----------------------------
def delaunay_edge_index(pos_xy: np.ndarray,
                        add_loops: bool = True,
                        make_undir: bool = True) -> torch.Tensor:
    """
    Build edge_index from 2D points using Delaunay triangulation.
    Falls back to fully-connected (without self loops) if triangulation fails (e.g., <3 points or collinear).
    """
    assert pos_xy.ndim == 2 and pos_xy.shape[1] == 2, "pos_xy must be (N,2)"
    N = pos_xy.shape[0]

    if N == 0:
        return torch.empty((2, 0), dtype=torch.long)

    edges = set()

    try:
        from scipy.spatial import Delaunay  # prefer scipy if available
        tri = Delaunay(pos_xy)
        simplices = tri.simplices  # (M,3)
        # each simplex contributes 3 undirected edges
        for a, b, c in simplices:
            edges.add(tuple(sorted((int(a), int(b)))))
            edges.add(tuple(sorted((int(b), int(c)))))
            edges.add(tuple(sorted((int(a), int(c)))))
    except Exception as e:
        # Fallback: fully-connected graph (no self loops)
        warnings.warn(f"[delaunay_edge_index] Delaunay failed ({e}). Falling back to fully-connected.")
        for i in range(N):
            for j in range(i + 1, N):
                edges.add((i, j))

    if make_undir:
        undirected = []
        for i, j in edges:
            undirected.append((i, j))
            undirected.append((j, i))
        edge_index = torch.tensor(undirected, dtype=torch.long).t().contiguous()
    else:
        directed = []
        for i, j in edges:
            directed.append((i, j))
        edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()

    if add_loops:
        # add self-loops
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_index, _ = add_self_loops(edge_index, num_nodes=N)

    return edge_index

# -----------------------------
# Sanity: spatial-only assertion
# -----------------------------
def assert_spatial_only(data: "Data"):
    """
    Ensure there is no temporal info inside this per-frame graph.
    If you carry a 'frame_id' tensor, assert it's constant.
    """
    # Example hooks if your Data has such attributes (commented by default):
    # if hasattr(data, "frame_id"):
    #     unique = torch.unique(data.frame_id)
    #     assert unique.numel() == 1, f"Temporal leakage: found multiple frame_ids {unique.tolist()}"
    # if hasattr(data, "dt") or hasattr(data, "vel"):
    #     raise AssertionError("Temporal features present in 1f-S baseline.")
    return

# -----------------------------
# Frameify: expand sequences
# -----------------------------
def frameify_all_frames(
    dataset_seq: Iterable[Any],
    get_num_frames: Callable[[Any], int],
    extractor: Callable[[Any, int], Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]],
    tip_num: Optional[int] = None,
    add_self_loops: bool = True,
    make_undirected: bool = True,
    require_delaunay: bool = True,
) -> List["Data"]:
    """
    Expand each sequence into per-frame PyG Data samples.
    extractor(seq_item, t) -> (pos_xy (N,2), x_node (N,F) or None, y (D))
    """
    samples: List[Data] = []
    for d in dataset_seq:
        T = int(get_num_frames(d))
        for t in range(T):
            pos_xy, x_node, y_label = extractor(d, t)

            # Optional node sub-selection (disabled by default to match teacher)
            if tip_num is not None and tip_num > 0 and pos_xy.shape[0] > tip_num:
                # Simple heuristic: take the first tip_num nodes (customize if you have a score)
                idx = np.arange(pos_xy.shape[0])[:tip_num]
                pos_xy = pos_xy[idx]
                if x_node is not None:
                    x_node = x_node[idx]

            # Build features tensor
            pos_t = torch.as_tensor(pos_xy, dtype=torch.float32)
            if x_node is None:
                x_t = pos_t  # default: use pos as node features
            else:
                x_t = torch.as_tensor(x_node, dtype=torch.float32)
                assert x_t.shape[0] == pos_t.shape[0], "x_node and pos must have same N"

            # Spatial-only edges
            if require_delaunay and pos_t.shape[0] >= 2:
                edge_index = delaunay_edge_index(pos_t.cpu().numpy(),
                                                 add_loops=add_self_loops,
                                                 make_undir=make_undirected)
            else:
                # If no Delaunay (e.g., N<2), create empty edges (self-loops can be added if needed)
                edge_index = torch.empty((2, 0), dtype=torch.long)
                if add_self_loops and pos_t.shape[0] > 0:
                    I = torch.arange(pos_t.shape[0], dtype=torch.long)
                    edge_index = torch.stack([I, I], dim=0)

            y_t = torch.as_tensor(y_label, dtype=torch.float32)
            data = Data(x=x_t, pos=pos_t, edge_index=edge_index, y=y_t)

            assert_spatial_only(data)
            samples.append(data)
    return samples

# -----------------------------
# Variant entry
# -----------------------------
def build_1f_S_datasets(
    train_dataset_raw: Iterable[Any],
    val_dataset_raw: Iterable[Any],
    get_num_frames: Callable[[Any], int],
    extractor: Callable[[Any, int], Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]],
    tip_num: Optional[int] = None,
    add_self_loops: bool = True,
    make_undirected: bool = True,
    require_delaunay: bool = True,
) -> Tuple[List["Data"], List["Data"]]:
    """
    Build GCN-1f-S datasets (frameify-all-frames, spatial-only).
    """
    train_dataset_run = frameify_all_frames(train_dataset_raw, get_num_frames, extractor,
                                            tip_num=tip_num,
                                            add_self_loops=add_self_loops,
                                            make_undirected=make_undirected,
                                            require_delaunay=require_delaunay)
    val_dataset_run   = frameify_all_frames(val_dataset_raw,   get_num_frames, extractor,
                                            tip_num=tip_num,
                                            add_self_loops=add_self_loops,
                                            make_undirected=make_undirected,
                                            require_delaunay=require_delaunay)
    print("[Variant] RUN = GCN-1f-S (frameify-all-frames, spatial-only)")
    return train_dataset_run, val_dataset_run

# -----------------------------
# Metrics helper (physical MAE)
# -----------------------------
def compute_mae_phys(pred_phys: "torch.Tensor", gt_phys: "torch.Tensor") -> "torch.Tensor":
    """
    Return per-dimension MAE in physical units (e.g., mm, degree).
    Shapes: [B, D]
    """
    assert pred_phys.shape == gt_phys.shape, "Shape mismatch for MAE"
    with torch.no_grad():
        mae = (pred_phys - gt_phys).abs().mean(dim=0)
    return mae

# Example inverse-transform hook (you should pass your own scaler)
def inverse_transform_y(y: "torch.Tensor", scaler) -> "torch.Tensor":
    """
    Apply your project's inverse transformation using the same scaler fitted on TRAIN targets.
    The `scaler` can be an object with attributes/behavior similar to sklearn's StandardScaler/MultiLabel.
    """
    if scaler is None:
        return y
    if hasattr(scaler, "inverse_transform"):
        # sklearn-like
        y_np = y.detach().cpu().numpy()
        inv = scaler.inverse_transform(y_np)
        return torch.as_tensor(inv, dtype=y.dtype, device=y.device)
    # Otherwise expect dict with 'mean' and 'std' (or 'scale')
    mean = getattr(scaler, "mean_", None) or getattr(scaler, "mean", None)
    std  = getattr(scaler,  "scale_", None) or getattr(scaler,  "std",  None)
    if mean is None or std is None:
        raise ValueError("Unsupported scaler type; provide an object with inverse_transform or mean_/scale_.")
    mean_t = torch.as_tensor(mean, dtype=y.dtype, device=y.device)
    std_t  = torch.as_tensor(std,  dtype=y.dtype, device=y.device)
    return y * std_t + mean_t
