# 🚀 LIANA批量测试快速开始指南

## 📁 文件列表

在 `scripts/` 目录下，你现在有以下脚本：

```
scripts/
├── run_all_liana_methods.sh    # 主脚本：运行5个数据集×8个方法
├── quick_test.sh                # 快速测试：验证环境是否正常
└── README_BATCH_RUN.md          # 完整文档：详细使用说明
```

---

## ⚡ 三步快速开始

### 步骤1: 快速测试（5-10分钟）

```bash
cd /root/autodl-tmp/lianaplus_manuscript-main/notebooks/classification/scripts

# 运行快速测试，确保一切正常
bash quick_test.sh
```

**预期输出**:
```
╔════════════════════════════════════════════════════════════╗
║                    测试成功！✓                              ║
╚════════════════════════════════════════════════════════════╝

耗时: 8分32秒
✓ 结果文件已生成
✓ 脚本工作正常，可以运行完整批量测试
```

---

### 步骤2: 运行批量测试（15-30小时）

#### 选项A: 使用screen（推荐）

```bash
# 创建screen会话
screen -S liana_batch

# 运行批量测试
bash run_all_liana_methods.sh

# 分离screen: 按 Ctrl+A，然后按 D
# 重新连接: screen -r liana_batch
```

#### 选项B: 使用nohup

```bash
# 后台运行
nohup bash run_all_liana_methods.sh > batch_run.log 2>&1 &

# 查看进度
tail -f batch_run.log
```

---

### 步骤3: 查看结果

```bash
# 查看完成的任务
ls -lh ../code/pipeline/output/*.csv | wc -l

# 查看批量报告
cat ../logs/batch_runs/batch_report_*.txt | tail -1

# 查看特定结果
head ../code/pipeline/output/kuppe_cellphonedb_det.csv
```

---

## 📊 测试矩阵

### 数据集（5个）
| 名称 | 描述 | 样本数 | 预计时间/方法 |
|------|------|--------|--------------|
| carraro | 肺 | ~16 | 15-20分钟 |
| velmeshev | 大脑（自闭症） | ~38 | 20-25分钟 |
| kuppe | 心脏 | ~78 | 25-30分钟 |
| habermann | 肺纤维化 | ~176 | 30-35分钟 |
| reichart | 心脏（扩张型心肌病） | ~171 | 35-45分钟 |

### LIANA方法（8个）
1. CellPhoneDB
2. CellChat
3. scSeqComm
4. SingleCellSignalR
5. LogFC
6. RankAggregate
7. GeometricMean
8. Connectome

### 总计
- **总任务数**: 5 × 8 = **40次运行**
- **预计总时间**: **15-30小时**

---

## 🎯 自定义运行

### 只运行特定数据集

编辑 `run_all_liana_methods.sh`，修改第34行：

```bash
DATASETS=(
    "carraro"      # ✓ 保留
    # "kuppe"      # ✗ 注释掉
    # "reichart"   # ✗ 注释掉
    # "habermann"  # ✗ 注释掉
    # "velmeshev"  # ✗ 注释掉
)
```

### 只运行特定方法

编辑 `run_all_liana_methods.sh`，修改第40行：

```bash
METHODS=(
    "cellphonedb"        # ✓ 保留
    "cellchat"           # ✓ 保留
    # "scseqcomm"        # ✗ 注释掉
    # "singlecellsignalr" # ✗ 注释掉
    # ...其他方法
)
```

---

## 📈 监控进度

### 实时监控

```bash
# 方法1: 查看日志
tail -f ../logs/batch_runs/*.log | grep -E "成功|失败|运行"

# 方法2: 计算完成数
watch -n 30 'ls ../code/pipeline/output/*.csv | wc -l'

# 方法3: 查看当前任务
ps aux | grep python | grep simplified_pipeline
```

### 查看统计

```bash
# 成功的任务
ls ../code/pipeline/output/*.csv | wc -l

# 查看最新报告
cat ../logs/batch_runs/batch_report_*.txt | tail
```

---

## ⚠️ 重要提示

### ✅ 数据泄露已修复
- 降维在CV循环内进行
- 测试集从未"见过"训练数据
- 结果可以安全发表

### 💾 磁盘空间
- 确保至少有 **20GB** 可用空间
- 结果文件: ~1-10MB/任务
- 日志文件: ~50-100MB/任务

### 🔄 断点续传
- 已完成的任务自动跳过
- 可以安全中断并重新运行
- 删除结果CSV可以重新运行该任务

---

## 🐛 常见问题

### Q1: "ModuleNotFoundError: No module named 'liana'"

```bash
# 确保激活正确环境
conda activate LIANA

# 检查安装
python -c "import liana; print('OK')"
```

### Q2: "内存不足 / OOM"

```bash
# 检查内存
free -h

# 如果内存不足，可以：
# 1. 关闭其他程序
# 2. 使用更小的数据集
# 3. 减少并行进程
```

### Q3: 如何暂停/继续？

```bash
# 如果使用screen:
# 暂停: Ctrl+A 然后 D
# 继续: screen -r liana_batch

# 如果使用nohup:
# 查找进程ID
ps aux | grep run_all_liana_methods

# 暂停（不推荐，因为会中断当前任务）
kill -STOP [PID]

# 继续
kill -CONT [PID]
```

---

## 📞 获取帮助

如遇问题，检查以下文件：

1. **日志文件**: `../logs/batch_runs/[dataset]_[method]_*.log`
2. **批量报告**: `../logs/batch_runs/batch_report_*.txt`
3. **详细文档**: `README_BATCH_RUN.md`

---

## 🎉 成功标志

当看到以下输出时，表示批量测试已完成：

```
╔════════════════════════════════════════════════════════════╗
║                      批量运行完成                          ║
╚════════════════════════════════════════════════════════════╝

统计信息:
  总任务数: 40
  成功: 40  ← 所有任务都成功
  失败: 0
  跳过: 0
```

---

## 📝 下一步

完成批量测试后，你可以：

1. **分析结果**: 比较不同方法的性能
2. **可视化**: 创建AUROC对比图表
3. **发表**: 结果已修复数据泄露，可以安全发表

---

**创建日期**: 2026-01-30
**版本**: 1.0
**状态**: ✅ 数据泄露已修复
