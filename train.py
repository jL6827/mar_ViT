# Updated train.py — 支持传入固定的训练/测试 CSV 文件（例如 processed_data_mean_train.csv / processed_data_mean_test.csv）
# 如果提供 --train_csv 与 --test_csv，程序将直接使用这两个文件（不再按日期随机切分）。
# 否则保持原有按日期随机划分（默认从单一 CSV 中按 train_frac 划分）。
#
# 用法示例（使用你预先分好的文件，80/20）：
# python train.py --train_csv processed_data_mean_train.csv --test_csv processed_data_mean_test.csv --config config.yaml --device cuda --epochs 60 --batch_size 8 --fp16

import os
import argparse
import yaml
import random
import numpy as np
import torch
from torch import nn, optim, amp
from torch.utils.data import DataLoader

from dataset import DailyDataset, collate_fn
from model import TokenTransformer
from utils import train_val_split_indices, ensure_dir, save_eval_per_var, save_eval_per_day

from tqdm import tqdm

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cfg_get(cfg, key_path, typ, default=None):
    parts = key_path.split(".")
    obj = cfg
    try:
        for p in parts:
            obj = obj[p]
    except Exception:
        return default
    if obj is None:
        return default
    try:
        return typ(obj)
    except Exception:
        try:
            s = str(obj).strip()
            s = s.replace(",", "")
            return typ(s)
        except Exception:
            return default

def evaluate_model(model, dataloader, device):
    model.eval()
    preds_by_day = {}
    targets_by_day = {}
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in dataloader:
            feats = batch["feats"].to(device)    # [B,T,F]
            targets = batch["targets"].to(device) # [B,T,4]
            mask = batch["mask"].to(device)      # [B,T]
            dates = batch["date"]
            outputs = model(feats, mask)         # [B,T,4]
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            mask_np = mask.cpu().numpy()
            for i, d in enumerate(dates):
                m = mask_np[i].astype(bool)
                if m.sum() == 0:
                    continue
                pvec = outputs_np[i, m, :]
                tvec = targets_np[i, m, :]
                preds_by_day[d] = pvec
                targets_by_day[d] = tvec
                all_preds.append(pvec)
                all_targets.append(tvec)
    if len(all_preds) == 0:
        return None, None, None
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    return preds_by_day, targets_by_day, (all_preds, all_targets)

def train(args):
    # 读取配置文件（显式指定编码，避免 Windows 下编码问题）
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    seed = cfg_get(cfg, "training.seed", int, 42)
    seed_everything(seed)
    ensure_dir("output")

    max_tokens = cfg_get(cfg, "data.max_tokens_per_sample", int, 200)
    feat_dim = cfg_get(cfg, "data.feat_dim", int, 8)
    embed_dim = cfg_get(cfg, "model.embed_dim", int, 128)
    n_layers = cfg_get(cfg, "model.n_layers", int, 4)
    n_heads = cfg_get(cfg, "model.n_heads", int, 4)
    mlp_dim = cfg_get(cfg, "model.mlp_dim", int, 256)
    dropout = cfg_get(cfg, "model.dropout", float, 0.1)

    train_frac = cfg_get(cfg, "training.train_frac", float, 0.85)
    lr = cfg_get(cfg, "training.lr", float, 1e-4)
    weight_decay = cfg_get(cfg, "training.weight_decay", float, 1e-5)
    cfg_epochs = cfg_get(cfg, "training.epochs", int, 60)

    # ---------- 这里是修改点：支持外部提供固定训练/测试 CSV ----------
    if args.train_csv is not None and args.test_csv is not None:
        # 使用用户提供的固定训练和测试文件
        print(f"Using provided train CSV: {args.train_csv}")
        print(f"Using provided test  CSV: {args.test_csv}")
        train_ds = DailyDataset(args.train_csv, max_tokens=max_tokens)
        val_ds = DailyDataset(args.test_csv, max_tokens=max_tokens)
        # 如果希望使用训练集统计量（如 lat/ lon/ depth / target）来标准化测试集，请把训练集的统计复制给测试集：
        # 只有当 dataset.py 中定义了这些属性时才复制（防护性赋值）
        attr_names = ["lat_mean", "lat_std", "lon_mean", "lon_std", "depth_mean", "depth_std",
                      "target_mean", "target_std"]
        for a in attr_names:
            if hasattr(train_ds, a) and hasattr(val_ds, a):
                setattr(val_ds, a, getattr(train_ds, a))
    else:
        # 原始行为：从单个 CSV 中按日期随机划分（train_frac）
        full_ds = DailyDataset(args.data, max_tokens=max_tokens)
        n = len(full_ds)
        train_idx, val_idx = train_val_split_indices(n, train_frac=train_frac, seed=seed)
        train_ds = DailyDataset(args.data, max_tokens=max_tokens, indices=train_idx)
        val_ds = DailyDataset(args.data, max_tokens=max_tokens, indices=val_idx)

    # batch_size / epochs from args override config defaults
    batch_size = args.batch_size if args.batch_size is not None else cfg_get(cfg, "runtime.batch_size", int, 8)
    epochs = args.epochs if args.epochs is not None else cfg_epochs

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = TokenTransformer(
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        max_tokens=max_tokens,
        out_dim=4
    ).to(device)

    # 确保 lr/weight_decay 是数值
    try:
        lr = float(lr)
    except Exception:
        lr = 1e-4
    try:
        weight_decay = float(weight_decay)
    except Exception:
        weight_decay = 1e-5

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.L1Loss(reduction='none')
    scaler = amp.GradScaler(enabled=(args.fp16 and device.type == "cuda"))

    best_val_loss = float('inf')
    log_rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", ncols=120)
        for batch in pbar:
            feats = batch["feats"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad()

            with amp.autocast(device_type='cuda', enabled=(args.fp16 and device.type == "cuda")):
                outputs = model(feats, mask)
                loss_all = criterion(outputs, targets)  # [B,T,4]
                loss_masked = loss_all.mean(dim=-1) * mask  # [B,T]
                loss = loss_masked.sum() / (mask.sum() + 1e-9)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=running_loss / n_batches)
        train_loss = running_loss / max(1, n_batches)

        # validation
        preds_by_day, targets_by_day, all_pair = evaluate_model(model, val_loader, device)
        if all_pair is None:
            val_loss = float('inf')
        else:
            all_preds, all_targets = all_pair
            val_loss = np.abs(all_preds - all_targets).mean()
            save_eval_per_day(preds_by_day, targets_by_day, "output/eval_per_day.csv")
            save_eval_per_var(all_preds, all_targets, "output/eval_per_var.csv")
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        import pandas as pd
        pd.DataFrame(log_rows).to_csv("output/train_val_log.csv", index=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cfg": cfg
            }, "output/model_best.pt")
        print(f"Epoch {epoch}: train_loss={train_loss:.6e}, val_loss={val_loss:.6e}, best_val={best_val_loss:.6e}")

    print("Training finished. Best val loss:", best_val_loss)
    print("Outputs saved in output/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="processed_data_mean.csv", help="如果不提供 train/test 文件，程序会从此文件按日期划分")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--train_csv", type=str, default=None, help="可选：指定训练文件（已预先划分好的 CSV）")
    parser.add_argument("--test_csv", type=str, default=None, help="可选：指定测试文件（已预先划分好的 CSV）")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true", help="Use AMP mixed precision (recommended for GPU)")
    args = parser.parse_args()

    with open(args.config, "r", encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    train(args)