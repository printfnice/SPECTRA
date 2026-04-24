# LIANA方法批量测试脚本使用说明

## 📋 概述

这个脚本可以在5个数据集上批量运行8个LIANA方法，使用修复数据泄露后的DET降维方法。

## 🎯 测试配置

### 数据集（5个）
1. **kuppe** - 心脏数据集
2. **reichart** - 心脏数据集（扩张型心肌病）
3. **habermann** - 肺纤维化数据集
4. **velmeshev** - 大脑数据集（自闭症）
5. **carraro** - 肺数据集

### LIANA方法（8个）
1. **CellPhoneDB** (`cellphonedb`) - 基于配体受体数据库
2. **CellChat** (`cellchat`) - 基于配体受体数据库
3. **scSeqComm** (`scseqcomm`) - 序列通讯方法
4. **SingleCellSignalR** (`singlecellsignalr`) - 单细胞信号方法
5. **LogFC** (`logfc`) - 基于差异表达
6. **RankAggregate** (`rank_aggregate`) - 排名聚合
7. **GeometricMean** (`geometric_mean`) - 几何平均
8. **Connectome** (`connectome`) - 连接组方法

### 降维方法
- **DET** (Deep Ensemble Transformers) - 已修复数据泄露

## 🚀 使用方法

### 基本用法

```bash
# 进入脚本目录
cd /root/autodl-tmp/lianaplus_manuscript-main/notebooks/classification/scripts

# 运行所有测试（5个数据集 × 8个方法 = 40次运行）
bash run_all_liana_methods.sh
```

### 后台运行

```bash
# 在后台运行并记录输出
nohup bash run_all_liana_methods.sh > batch_run.log 2>&1 &

# 查看进度
tail -f batch_run.log

# 或使用更友好的监控
watch -n 5 'tail -20 batch_run.log'
```

### 使用screen（推荐）

```bash
# 创建新的screen会话
screen -S liana_batch

# 运行脚本
bash run_all_liana_methods.sh

# 分离screen: Ctrl+A 然后按 D
# 重新连接: screen -r liana_batch
```

## 📊 输出文件

### 结果文件位置
```
notebooks/classification/code/pipeline/output/
├── kuppe_cellphonedb_det.csv
├── kuppe_cellchat_det.csv
├── ...
├── carraro_geometric_mean_det.csv
└── carraro_connectome_det.csv
```

### 日志文件位置
```
notebooks/classification/logs/batch_runs/
├── kuppe_cellphonedb_det_20260130_123456.log
├── batch_report_20260130_123456.txt
└── ...
```

### 结果CSV格式
每个结果文件包含：
- `dataset`: 数据集名称
- `score_key`: LIANA方法名称
- `reduction_name`: 降维方法（det）
- `state`: 随机种子编号
- `fold`: 交叉验证fold编号
- `auroc`: AUROC分数
- `f1_score`: F1分数
- `oob_score`: OOB分数

## ⏱️ 预计运行时间

### 单次运行时间估算
- **快速数据集** (carraro, velmeshev): ~15-25分钟/方法
- **中等数据集** (kuppe, habermann): ~25-35分钟/方法
- **大型数据集** (reichart): ~35-45分钟/方法

### 总时间估算
- **总任务数**: 40次运行
- **预计总时间**: 15-30小时（取决于服务器性能）
- **建议**: 使用screen或nohup后台运行

## 🔧 自定义配置

### 修改数据集列表

编辑脚本中的 `DATASETS` 数组：

```bash
# 只运行部分数据集
DATASETS=(
    "kuppe"
    "reichart"
    # "habermann"  # 注释掉不需要的
    # "velmeshev"
    # "carraro"
)
```

### 修改方法列表

编辑脚本中的 `METHODS` 数组：

```bash
# 只运行部分方法
METHODS=(
    "cellphonedb"
    "cellchat"
    # "scseqcomm"  # 注释掉不需要的
    # ...
)
```

### 跳过已完成的任务

脚本会自动检查输出文件是否存在：
- 如果结果文件已存在 → 自动跳过
- 如果要重新运行 → 删除对应的CSV文件

```bash
# 删除特定结果以重新运行
rm output/kuppe_cellphonedb_det.csv
```

## 📈 监控进度

### 实时查看进度

```bash
# 查看最新日志
tail -f logs/batch_runs/batch_run_*.log

# 查看成功完成的任务
ls -lh code/pipeline/output/*.csv | wc -l

# 查看当前运行的任务
ps aux | grep python | grep simplified_pipeline
```

### 检查失败任务

```bash
# 查看最新的报告文件
cat logs/batch_runs/batch_report_*.txt

# 查看失败日志
grep -l "失败\|Error" logs/batch_runs/*.log
```

## ⚠️ 注意事项

### 1. 数据泄露已修复
- ✅ 降维在CV循环内进行
- ✅ 测试集从未"见过"训练数据
- ✅ 结果可以发表

### 2. 磁盘空间
- 每个结果CSV约1-10MB
- 日志文件可能很大（50-100MB/任务）
- 确保有足够的磁盘空间（建议至少20GB）

### 3. 内存和GPU
- 大数据集（reichart）可能需要大量内存
- GPU会自动使用（如果可用）
- 如遇OOM错误，可以减少批次大小

### 4. 断点续传
- 已完成的任务会自动跳过
- 可以安全地中断并重新运行
- 脚本会从未完成的任务继续

## 🐛 故障排查

### 问题1: "ModuleNotFoundError"

```bash
# 确保在正确的conda环境
conda activate LIANA

# 检查环境
python -c "import liana; print('LIANA OK')"
```

### 问题2: "内存不足"

```bash
# 检查可用内存
free -h

# 如果内存不足，减少并行进程或使用更小的数据集
```

### 问题3: "结果文件为空"

```bash
# 检查日志文件
cat logs/batch_runs/[最新日志文件].log | tail -100

# 查看错误信息
grep -i error logs/batch_runs/[最新日志文件].log
```

## 📞 获取帮助

如遇问题，请检查：
1. 日志文件：`logs/batch_runs/*.log`
2. 报告文件：`logs/batch_runs/batch_report_*.txt`
3. 原始输出：如果使用nohup，检查`batch_run.log`

## 📝 示例：快速测试

如果要快速测试单个组合：

```bash
cd /root/autodl-tmp/lianaplus_manuscript-main/notebooks/classification/code/pipeline

# 修改simplified_pipeline_optimized.py中的配置
# dataset_name = 'kuppe'
# liana_method = 'cellphonedb'
# reduction_method = 'det'

python simplified_pipeline_optimized.py
```

## 🎯 成功标志

脚本成功完成时会显示：

```
╔════════════════════════════════════════════════════════════════════╗
║                      批量运行完成                                  ║
╚════════════════════════════════════════════════════════════════════╝

统计信息:
  总任务数: 40
  成功: 40
  失败: 0
  跳过: 0
  总耗时: XXX分钟
```

---

**创建时间**: 2026-01-30
**版本**: 1.0
**修复**: 数据泄露已修复 ✅
