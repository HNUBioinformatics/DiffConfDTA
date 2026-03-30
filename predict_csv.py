
# predict_csv.py
# -*- coding: utf-8 -*-
"""
用训练好的 best.pt 对一张自定义 CSV 批量预测。
关键：自定义 collate 对 smile_vec / protein_vec 做动态右侧 0-padding，
避免 PyG 在 batch 拼接时因长度不等报错。

用法：
python predict_csv.py --ckpt best.pt --csv your.csv --out results/preds_custom.csv --batch_size 256
"""

import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader as TorchDL
from torch_geometric.data import Batch

from modelfeaturefusionGCN import DTA_GCN
# 复用 CSV→(dm, dp) 的构造逻辑；来自 explain_from_csv.py
from explain_from_csv import load_csv


def pad_2d(x: torch.Tensor, L: int) -> torch.Tensor:
    """把 shape [1, l] 或 [l] 的向量右侧 0-padding 到 [1, L]。"""
    x = x.view(1, -1)
    l = x.size(1)
    if l >= L:
        return x[:, :L]
    out = torch.zeros(1, L, dtype=x.dtype, device=x.device)
    out[:, :l] = x
    return out


def collate_pad(batch):
    """
    batch: List[(dm, dp)]
    仅对 dm.smile_vec / dp.protein_vec 做长度对齐，其余图结构/节点特征保持原样。
    """
    dms, dps = zip(*batch)
    max_smiles = max(int(getattr(dm, "smile_vec").numel()) for dm in dms)
    max_prot   = max(int(getattr(dp, "protein_vec").numel()) for dp in dps)

    for dm in dms:
        dm.smile_vec = pad_2d(dm.smile_vec, max_smiles)
    for dp in dps:
        dp.protein_vec = pad_2d(dp.protein_vec, max_prot)

    batch_mol = Batch.from_data_list(list(dms))
    batch_pro = Batch.from_data_list(list(dps))
    return batch_mol, batch_pro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="best.pt 路径")
    ap.add_argument("--csv", required=True, help="包含 drug_smiles/compound_iso_smiles, target_sequence 的 CSV")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out", default="results/preds_custom.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("DTA_GCN Loading ...")
    model = DTA_GCN().to(device).eval()

    # 宽松加载：剥离 module. 前缀，允许 strict=False
    state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[load] missing:", len(missing), "unexpected:", len(unexpected))

    df, items = load_csv(args.csv)
    loader = TorchDL(items, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad)

    preds = []
    with torch.no_grad():
        for dm, dp in loader:
            dm, dp = dm.to(device), dp.to(device)
            out = model(dm, dp).view(-1)
            preds.extend(out.detach().cpu().tolist())

    out_df = df.copy()
    out_df["pred_affinity"] = preds
    if "affinity" in out_df.columns:
        out_df["residual"] = out_df["pred_affinity"] - out_df["affinity"]

    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print("[done]", args.out)


if __name__ == "__main__":
    main()
