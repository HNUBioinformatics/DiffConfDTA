import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from modelfeaturefusionGCN import DTA_GCN
from dataloader import create_DTA_dataset
from utils import collate, mse, rm2, rmse, ci, r2s, pearson, spearman

def load_state_dict_lenient(model: nn.Module, ckpt_path: str, device: torch.device):
    """尽量稳妥地加载：剥离 module. 前缀；跳过形状不匹配的键；给出加载报告。"""
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # 统一去掉 DataParallel 的 'module.' 前缀
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v

    # 只保留与当前模型同名且同shape的参数
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in new_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)

    msg = []
    if skipped:
        msg.append(f"[load_state] Skipped {len(skipped)} mismatched keys (e.g., {skipped[:5]})")
    missing = [k for k in model_state.keys() if k not in filtered]
    if missing:
        msg.append(f"[load_state] Missing {len(missing)} keys now randomly initialized (e.g., {missing[:5]})")

    model.load_state_dict(filtered, strict=False)
    print("\n".join(msg) if msg else "[load_state] All keys matched.")

def predict(model: nn.Module, device: torch.device, loader: DataLoader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for data in loader:
            data_mol, data_pro = data[0].to(device), data[1].to(device)
            out = model(data_mol, data_pro)      # eval 模式下返回单个张量 [B,1]
            y = data_mol.y.view(-1, 1).to(device)
            ys.append(y)
            ps.append(out)
    y_true = torch.cat(ys, dim=0).cpu().numpy().flatten()
    y_pred = torch.cat(ps, dim=0).cpu().numpy().flatten()
    return y_true, y_pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="kiba", choices=["davis", "kiba"])
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to model checkpoint (.pt/.pth). Use NEW checkpoint after code changes.")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--out_csv", type=str, default=None)
    args = parser.parse_args()

    print(f"\npredicting for test dataset using  DTA_GCN")
    print(f"dataset: {args.dataset}")

    # data
    _, test_data = create_DTA_dataset(args.dataset)
    print(f"test entries: {len(test_data)}")
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model
    model = DTA_GCN().to(device)
    print("DTA_GCN Loading ...")
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    load_state_dict_lenient(model, args.ckpt, device)

    # predict
    print(f"Make prediction for {len(test_data)} samples...")
    G, P = predict(model, device, test_loader)

    # metrics
    metrics = {
        "mse": mse(G, P),
        "rm2": rm2(G, P),
        "rmse": rmse(G, P),
        "ci": ci(G, P),
        "r2s": r2s(G, P),
        "pearson": pearson(G, P),
        "spearman": spearman(G, P),
    }
    rounded = {k: round(v, 3) for k, v in metrics.items()}
    print("dataset,model,mse,rm2,rmse,ci,r2s,pearson,spearman")
    print(args.dataset, "DTA_GCN", rounded["mse"], rounded["rm2"], rounded["rmse"],
          rounded["ci"], rounded["r2s"], rounded["pearson"], rounded["spearman"])

    # save csv
    os.makedirs("results", exist_ok=True)
    out_csv = args.out_csv or os.path.join("results", f"DTA_GCN_result_{args.dataset}.csv")
    header_needed = not os.path.isfile(out_csv)
    with open(out_csv, "a") as f:
        if header_needed:
            f.write("dataset,model,mse,rm2,rmse,ci,r2s,pearson,spearman\n")
        f.write(",".join(map(str, [
            args.dataset, "DTA_GCN",
            round(metrics["mse"], 6), round(metrics["rm2"], 6), round(metrics["rmse"], 6),
            round(metrics["ci"], 6), round(metrics["r2s"], 6),
            round(metrics["pearson"], 6), round(metrics["spearman"], 6),
        ])) + "\n")

if __name__ == "__main__":
    main()
