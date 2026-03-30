# -*-coding:utf-8-*-
import os, sys
import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

import numpy as np
import pandas as pd

from dataloader import create_DTA_dataset
# 关键：用你已修改过的模型文件
from modelfeaturefusionGCN import DTA_GCN

from utils import *
from log.train_logger import TrainLogger

# ---------- 训练与评测 ----------
def train_one_epoch(model, device, loader, optimizer, loss_fn, epoch):
    model.train()
    running = 0.0
    n = 0
    print(f"Training on {len(loader.dataset)} samples...")
    for data in loader:
        optimizer.zero_grad()
        data_mol = data[0].to(device)
        data_pro = data[1].to(device)

        out = model(data_mol, data_pro)
        # 关键：有的 forward 返回 (pred, aux...)，只拿第一个作为预测
        if isinstance(out, tuple):
            out = out[0]
        # 保险：确保形状为 [B, 1]
        out = out.view(-1, 1)

        y = data_mol.y.view(-1, 1).float().to(device)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        running += loss.item() * y.size(0)
        n += y.size(0)

    print(f"Train epoch: {epoch}\tLoss: {running / max(1, n):.6f}")

@torch.no_grad()
def evaluate(model, device, loader):
    model.eval()
    ys, ps = [], []
    print(f"Make prediction for {len(loader.dataset)} samples...")
    for data in loader:
        data_mol = data[0].to(device)
        data_pro = data[1].to(device)

        out = model(data_mol, data_pro)
        if isinstance(out, tuple):
            out = out[0]
        out = out.view(-1, 1)

        ys.append(data_mol.y.view(-1, 1).cpu())
        ps.append(out.cpu())

    y_true = torch.cat(ys, dim=0).numpy().flatten()
    y_pred = torch.cat(ps, dim=0).numpy().flatten()
    return y_true, y_pred

# ---------- 主程序 ----------
def main():
    parser = argparse.ArgumentParser()
    # 与原脚本等价：dataset 用 0/1 选择 davis/kiba；cuda 用 0/1 选择 cuda:0/cuda:1
    parser.add_argument("--dataset_id", type=int, default=1, choices=[0, 1], help="0=davis, 1=kiba")
    parser.add_argument("--cuda_id", type=int, default=0, help="GPU id, e.g., 0 -> cuda:0")
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.0005)
    args = parser.parse_args() if len(sys.argv) > 1 else parser.parse_args([])

    datasets = ['davis', 'kiba']
    dataset = datasets[args.dataset_id]
    cuda_name = f"cuda:{args.cuda_id}"

    params = dict(
        data_root="data",
        save_dir="save",
        dataset=dataset,
        save_model="save_model",
        lr=args.lr,
        batch_size=args.batch_size,
        model_name="DTA_GCN"
    )
    logger = TrainLogger(params)
    logger.info(__file__)
    print(f"\nrunning on DTA_GCN_{dataset}")

    # 数据
    train_data, test_data = create_DTA_dataset(dataset)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_data,  batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # 设备
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

    # 模型
    model = DTA_GCN().to(device)
    # 重要：形状自检，应该是 (1024, 256)
    print("[shape check] fc1 weight:", tuple(model.fc1.weight.shape))

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_mse = float("inf")
    best_epoch = -1

    # 保存目录（新目录，避免旧权重混入）
    run_dir = os.path.join("runs", f"{dataset}_DTAGCN_v2")
    os.makedirs(run_dir, exist_ok=True)
    best_ckpt_path = os.path.join(run_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, device, train_loader, optimizer, loss_fn, epoch)
        G, P = evaluate(model, device, test_loader)
        test_mse = mse(G, P)
        logger.info(f"epoch-{epoch}, mse-{test_mse:.4f}")

        if test_mse < best_mse:
            best_mse = test_mse
            best_epoch = epoch
            # 1) 继续保留你原 utils 的保存（如果你在别处用到）
            save_model_dict(model, logger.get_model_dir(), f"epoch-{epoch}, mse-{test_mse:.4f}")
            # 2) 另外保存一个“纯 state_dict”到固定路径，供 predict.py --ckpt 使用
            to_save = model.module.state_dict() if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model.state_dict()
            torch.save(to_save, best_ckpt_path)
            print(f"[save] rmse improved at epoch {best_epoch}; best_mse: {best_mse:.6f}")
            print(f"[save] wrote new checkpoint to {best_ckpt_path}")
        else:
            print(f"No improvement since epoch {best_epoch}; best_mse: {best_mse:.6f}")

    print("train success!")

if __name__ == "__main__":
    main()
