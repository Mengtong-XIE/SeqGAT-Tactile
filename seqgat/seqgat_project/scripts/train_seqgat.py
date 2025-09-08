
import os, argparse, math, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from models.seqgat import SeqGAT
from data.pt_dataset import load_dataset

def set_seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def mae_rmse(y_true, y_pred):
    with torch.no_grad():
        err = torch.abs(y_true - y_pred)
        return {
            "MAE_Y": err[:,0].mean().item(),
            "MAE_theta": err[:,1].mean().item(),
            "RMSE_Y": torch.sqrt(((y_true[:,0]-y_pred[:,0])**2).mean()).item(),
            "RMSE_theta": torch.sqrt(((y_true[:,1]-y_pred[:,1])**2).mean()).item()
        }

def train_one_epoch(model, loader, opt, device, wY=1.0, wTh=1.0):
    model.train(); loss_fn = nn.SmoothL1Loss(reduction='none'); tot=0.0; n=0
    for data in loader:
        data = data.to(device); y = data.y; p = model(data)
        lv = loss_fn(p, y); loss = wY*lv[:,0].mean() + wTh*lv[:,1].mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        tot += loss.item() * y.size(0); n += y.size(0)
    return tot / max(n,1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ys=[]; ps=[]
    for data in loader:
        data = data.to(device); ys.append(data.y); ps.append(model(data))
    y = torch.cat(ys,0); p = torch.cat(ps,0); return mae_rmse(y,p)

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.get("seed",0))

    dataset = load_dataset(cfg["data_root"], glob_pattern=cfg.get("glob","*.pt"), field_map=cfg.get("field_map",{}), tip_num=cfg.get("tip_num",None), k_last_frames=cfg.get("k_last_frames",None))
    print(f"Loaded {len(dataset)} samples from {cfg['data_root']} (k_last_frames={cfg.get('k_last_frames',None)}, tip_num={cfg.get('tip_num',None)})")
    n_total = len(dataset); n_val = int(n_total * cfg.get("val_ratio",0.2)); n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.get("seed",0)))

    train_loader = DataLoader(train_set, batch_size=cfg.get("batch_size",8), shuffle=True,  num_workers=cfg.get("num_workers",0))
    val_loader   = DataLoader(val_set,   batch_size=cfg.get("batch_size",8), shuffle=False, num_workers=cfg.get("num_workers",0))

    model = SeqGAT(in_dim=cfg["in_dim"], hidden_dim=cfg.get("hidden_dim",128), num_layers=cfg.get("num_layers",4), heads=cfg.get("heads",4),
                   edge_dim_s=cfg.get("edge_dim_s",0), edge_dim_t=cfg.get("edge_dim_t",0), time_emb_dim=cfg.get("time_emb_dim",32),
                   dropout=cfg.get("dropout",0.1), readout=cfg.get("readout","meanmax")).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr",3e-4), weight_decay=cfg.get("weight_decay",1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.get("epochs",100))

    best = math.inf; best_metrics=None; key = cfg.get("early_stop_key","MAE_Y")
    for epoch in range(1, cfg.get("epochs",100)+1):
        tr = train_one_epoch(model, train_loader, opt, device, *cfg.get("loss_weights",[1.0,1.0]))
        metrics = evaluate(model, val_loader, device); sched.step()
        score = metrics[key]
        if score < best:
            best = score; best_metrics = metrics
            if cfg.get("save_ckpt",True):
                os.makedirs(cfg.get("out_dir","./outputs"), exist_ok=True)
                torch.save({"model":model.state_dict(), "cfg":cfg, "metrics":metrics}, os.path.join(cfg.get("out_dir","./outputs"), "seqgat_best.pth"))
        print(f"Ep{epoch:03d} | loss={tr:.4f} | MAE_Y={metrics['MAE_Y']:.4f} | MAE_th={metrics['MAE_theta']:.4f} | RMSE_Y={metrics['RMSE_Y']:.4f} | RMSE_th={metrics['RMSE_theta']:.4f}")
    print('Best:', best_metrics)

if __name__ == "__main__":
    import yaml
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=str, default=os.path.join("configs","seqgat_k5_st.yaml"))
    args = ap.parse_args(); cfg = yaml.safe_load(open(args.config,"r")); main(cfg)
