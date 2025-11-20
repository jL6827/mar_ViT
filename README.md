```markdown
# Transformer-based (ViT-like) model for 4D ocean variable prediction

说明（中文）：

这个项目实现了一个基于 Transformer（借鉴 ViT 的 token/patch 思路）的端到端数据驱动模型，用以从每个观测时刻的一组点（经度、纬度、深度、时间）预测同一组点上的四个物理量：uo、vo、so、thetao。

特点：
- 不使用你给出的微分方程模型中的任何内容（纯数据驱动）。
- 将同一天的所有观测点当作一个“样本”，把每个观测点当作 transformer 的一个 token（类似 ViT 的 patch tokens）。
- 支持可变长度日内观测点，使用 padding 和 mask。
- 训练/验证使用 MAE（L1）损失，并在训练结束后输出：
  - output/eval_per_day.csv：按日期统计的 MAE（类似你给的格式）
  - output/eval_per_var.csv：按变量（so, thetao, uo, vo）统计的总体 MAE
  - output/model_best.pt：保存的最佳模型权重
- 训练/测试切分：默认训练集 85%，测试集 15%（基于日期划分）。
- 推荐显卡：已用较小模型和可选 fp16 混合精度，适配 RTX 4060 Laptop 8GB。根据显存，可调小 batch_size（推荐 4 或 8）。

输入数据：
- 请将你的数据（示例中为 processed_data_mean.csv）放到项目根目录，csv 中需要包含至少以下列（列名区分大小写）：
  - time 或 date（时间戳，如 2023/3/1 或 2023-03-01 12:00:00，程序会解析日期）
  - segment_id（可选，用于排序）
  - latitude, longitude, depth
  - so, thetao, uo, vo （作为 target）
如果列名与上述稍有不同，请在 train.py 中做小修改或在 dataset.py 中添加映射。

快速开始（建议在 conda / venv 中）：

1. 安装依赖：
   pip install -r requirements.txt

2. 训练（示例）：
   # 使用 GPU（默认自动选 cuda，如需强制 CPU 请传 --device cpu）
   python train.py --data processed_data_mean.csv --config config.yaml --device cuda --epochs 60 --batch_size 8 --fp16

   如果出现显存不足（OOM），将 --batch_size 调小到 4 或 2；或者去掉 --fp16。

3. 训练结束后，检查 output/ 目录，里面包含：
   - model_best.pt
   - eval_per_day.csv
   - eval_per_var.csv
   - train_val_log.csv（可选训练过程记录）

说明与可调参数：
- config.yaml 中包含模型尺寸、学习率、pad 最大序列长度等超参数。
- 模型是一个轻量级 transformer（embed_dim 默认 128，layers 默认 4），你可以增减层数和宽度以平衡精度与显存。

如果你需要我把模型改成更像官方 ViT（把空间网格展开为 patch），或希望输入包含历年同日的历史序列，请告诉我你的 processed_data_mean.csv 的精确结构（头几行），我可以针对性调整数据处理与模型结构。
```