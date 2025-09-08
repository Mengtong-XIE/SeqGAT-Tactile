
import argparse, json, torch
from data.pt_dataset import load_dataset
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--glob", default="*.pt")
    p.add_argument("--tip_num", type=int, default=331)
    p.add_argument("--k_last_frames", type=int, default=5)
    p.add_argument("--field_map", default="")
    args = p.parse_args()
    fmap = json.loads(args.field_map) if args.field_map else {}
    ds = load_dataset(args.data_root, glob_pattern=args.glob, field_map=fmap, tip_num=args.tip_num, k_last_frames=args.k_last_frames)
    print("Total samples:", len(ds))
    if len(ds)>0:
        d0 = ds[0]
        def shp(x): 
            return None if x is None else (tuple(x.shape), str(x.dtype))
        print("x:", shp(d0.x))
        print("y:", shp(d0.y))
        print("t:", shp(d0.t), "min/max", int(d0.t.min()), int(d0.t.max()))
        print("edge_index_s:", shp(getattr(d0, "edge_index_s", None)))
        print("edge_index_t:", shp(getattr(d0, "edge_index_t", None)))
        print("edge_attr_t:", shp(getattr(d0, "edge_attr_t", None)))
