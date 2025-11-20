# train.py (修改版：安全读取 config 值并显式转换类型，避免从 YAML 中得到字符串导致错误)
import os
import argparse
import yaml
import random
import numpy as np
import torch
from torch import nn, optim
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
    """
    Get nested value from cfg by dot path, cast to typ, fallback to default.
    Example: cfg_get(cfg, "training.lr", float, 1e-4)
    """
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
        # try converting from string with possible commas or spaces
        try:
            s = str(obj).strip()
            # remove thousand separators
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
    # config
    # 显式指定 encoding='utf-8' 以避免 Windows 上的编码问题
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 从 cfg 安全获取各类参数并做类型转换
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

    # Dataset split
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
    if not isinstance(lr, float) and not isinstance(lr, int):
        try:
            lr = float(lr)
        except Exception:
            lr = 1e-4
    if not isinstance(weight_decay, float) and not isinstance(weight_decay, int):
        try:
            weight_decay = float(weight_decay)
        except Exception:
            weight_decay = 1e-5

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.L1Loss(reduction='none')  # we'll mask manually

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

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
            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
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
            # save per-day and per-var
            save_eval_per_day(preds_by_day, targets_by_day, "output/eval_per_day.csv")
            save_eval_per_var(all_preds, all_targets, "output/eval_per_var.csv")
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        # save training log
        import pandas as pd
        pd.DataFrame(log_rows).to_csv("output/train_val_log.csv", index=False)

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cfg": cfg
            }, "output/model_best.pt")
        print(f"Epoch {epoch}: train_loss={train_loss:.6e}, val_loss={val_loss:.6e}, best_val={best_val_loss:.6e}")

    # final evaluation on validation set (already saved)
    print("Training finished. Best val loss:", best_val_loss)
    print("Outputs saved in output/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="processed_data_mean.csv")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true", help="Use AMP mixed precision (recommended for GPU)")
    args = parser.parse_args()

    # load config then override (we still read it here to provide defaults if needed)
    with open(args.config, "r", encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    # if args.epochs or batch_size not provided they'll be resolved inside train()
    train(args)