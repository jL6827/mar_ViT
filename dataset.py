import os
import math
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import Dataset

def parse_time_to_features(t):
    # t can be string or pandas.Timestamp
    # return time_of_day sin/cos and day_of_year sin/cos
    if isinstance(t, str):
        try:
            dt = pd.to_datetime(t)
        except:
            # try different format
            dt = datetime.strptime(t, "%Y/%m/%d")
    else:
        dt = pd.to_datetime(t)
    sec_of_day = dt.hour * 3600 + dt.minute * 60 + dt.second
    day_frac = sec_of_day / 86400.0
    day_sin = math.sin(2 * math.pi * day_frac)
    day_cos = math.cos(2 * math.pi * day_frac)
    day_of_year = dt.timetuple().tm_yday / 365.0
    doy_sin = math.sin(2 * math.pi * day_of_year)
    doy_cos = math.cos(2 * math.pi * day_of_year)
    return day_sin, day_cos, doy_sin, doy_cos

class DailyDataset(Dataset):
    """
    每个 sample 是一天（或一个 timestamp）内的所有观测点（tokens）。
    CSV 需要包含列：time/date, segment_id (可选), latitude, longitude, depth, so, thetao, uo, vo
    """
    def __init__(self, csv_path, max_tokens=200, min_tokens=1, mode="train", indices=None):
        df = pd.read_csv(csv_path)
        # 尝试识别时间列
        time_col = None
        for c in ["time", "date", "datetime", "timestamp"]:
            if c in df.columns:
                time_col = c
                break
        if time_col is None:
            raise ValueError("CSV 中未找到时间列，请提供 time/date/datetime 列。")
        df[time_col] = pd.to_datetime(df[time_col])
        # 标准列名要求
        require_cols = ["latitude", "longitude", "depth", "so", "thetao", "uo", "vo"]
        for c in require_cols:
            if c not in df.columns:
                raise ValueError(f"CSV 缺少必要列: {c}")

        # group by date (只按日期分组，若你需要按完整时间戳分组可改)
        df["date_only"] = df[time_col].dt.date
        groups = df.groupby("date_only")
        samples = []
        for date_val, g in groups:
            g = g.sort_values(by=["segment_id"] if "segment_id" in g.columns else g.columns[0])
            samples.append((date_val, g.reset_index(drop=True)))
        # 如果 indices 指定，使用它（用于 train/test 划分）
        if indices is not None:
            samples = [samples[i] for i in indices]

        self.samples = samples
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

        # 对经度/纬度/depth 进行简单的标准化（基于整个文件）
        lat_all = df["latitude"].values
        lon_all = df["longitude"].values
        depth_all = df["depth"].values
        # 保存全局 stats
        self.lat_mean, self.lat_std = lat_all.mean(), lat_all.std() + 1e-9
        self.lon_mean, self.lon_std = lon_all.mean(), lon_all.std() + 1e-9
        self.depth_mean, self.depth_std = depth_all.mean(), depth_all.std() + 1e-9

    def __len__(self):
        return len(self.samples)

    def _build_features_targets(self, g):
        # g: dataframe for one day
        n = len(g)
        use_n = min(n, self.max_tokens)
        # features per token:
        # [lat_norm, lon_norm, depth_norm, seg_id_norm, time_sin, time_cos, doy_sin, doy_cos] -> feat_dim=8
        feats = np.zeros((self.max_tokens, 8), dtype=np.float32)
        targets = np.zeros((self.max_tokens, 4), dtype=np.float32)  # so, thetao, uo, vo
        mask = np.zeros((self.max_tokens,), dtype=np.float32)
        for i in range(use_n):
            row = g.iloc[i]
            lat = (row["latitude"] - self.lat_mean) / self.lat_std
            lon = (row["longitude"] - self.lon_mean) / self.lon_std
            depth = (row["depth"] - self.depth_mean) / self.depth_std
            seg_id = row["segment_id"] if "segment_id" in row.index else i
            # normalize seg id roughly into [0,1]
            seg_id_norm = seg_id / max(1, use_n)
            t = row.get("time", row.get("date", None))
            # some csv used full timestamp or date; try both
            if isinstance(t, pd.Timestamp):
                tval = t
            else:
                # try to get from any available column in row
                # fallback: use index as time
                try:
                    tval = row["time"]
                except:
                    tval = pd.Timestamp(g.iloc[0]["time"])
            time_sin, time_cos, doy_sin, doy_cos = parse_time_to_features(tval)
            feats[i, :] = np.array([lat, lon, depth, seg_id_norm, time_sin, time_cos, doy_sin, doy_cos], dtype=np.float32)
            targets[i, :] = np.array([row["so"], row["thetao"], row["uo"], row["vo"]], dtype=np.float32)
            mask[i] = 1.0
        return feats, targets, mask

    def __getitem__(self, idx):
        date_val, g = self.samples[idx]
        feats, targets, mask = self._build_features_targets(g)
        # convert to tensors
        return {
            "date": str(date_val),
            "feats": torch.from_numpy(feats),         # [max_tokens, feat_dim]
            "targets": torch.from_numpy(targets),     # [max_tokens, 4]
            "mask": torch.from_numpy(mask)            # [max_tokens]
        }

def collate_fn(batch):
    # batch is a list of dicts
    dates = [b["date"] for b in batch]
    feats = torch.stack([b["feats"] for b in batch], dim=0)       # [B, T, feat_dim]
    targets = torch.stack([b["targets"] for b in batch], dim=0)   # [B, T, 4]
    mask = torch.stack([b["mask"] for b in batch], dim=0)         # [B, T]
    return {"date": dates, "feats": feats, "targets": targets, "mask": mask}