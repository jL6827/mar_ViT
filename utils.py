import os
import csv
import math
import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def train_val_split_indices(n_samples, train_frac=0.85, seed=42):
    indices = list(range(n_samples))
    train_idx, val_idx = train_test_split(indices, train_size=train_frac, random_state=seed, shuffle=True)
    return train_idx, val_idx

def save_eval_per_var(all_preds, all_targets, out_path):
    # all_preds, all_targets: concatenated numpy arrays [N_tokens_total, 4] (matching mask filtered)
    # variables order: so, thetao, uo, vo
    diff = np.abs(all_preds - all_targets)
    mae_vars = diff.mean(axis=0)
    df = pd.DataFrame({
        "variable": ["so", "thetao", "uo", "vo"],
        "mae": mae_vars.tolist(),
        "count": [len(all_targets)]*4
    })
    df.to_csv(out_path, index=False)

def save_eval_per_day(preds_by_day, targets_by_day, out_path):
    # preds_by_day, targets_by_day: dict date_str -> ndarray [n_tokens_day, 4]
    rows = []
    for date, p in preds_by_day.items():
        t = targets_by_day[date]
        assert p.shape == t.shape
        diff = np.abs(p - t)
        mae_so = diff[:,0].mean()
        mae_thetao = diff[:,1].mean()
        mae_uo = diff[:,2].mean()
        mae_vo = diff[:,3].mean()
        mae_overall = diff.mean()
        rows.append({"date": date, "mae_so": mae_so, "mae_thetao": mae_thetao, "mae_uo": mae_uo, "mae_vo": mae_vo, "mae_overall": mae_overall})
    df = pd.DataFrame(rows)
    df = df.sort_values("date")
    df.to_csv(out_path, index=False)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)