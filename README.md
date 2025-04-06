# 细胞图像增强效果评估实验

本项目使用YOLOv8评估不同增强策略对细胞图像检测与分割任务的性能影响。

## 目录结构

```
├── data_original.yaml    # 原始数据集配置
├── data_a.yaml           # 增强100张数据集配置
├── data_b.yaml           # 增强500张数据集配置
├── data_c.yaml           # 增强1000张数据集配置
├── setup_yolov8.sh       # 环境安装脚本
├── prepare_data.py       # 数据准备脚本
├── train_eval.py         # 训练评估脚本
├── original/             # 原始数据集
├── a/                    # 增强100张数据集
├── b/                    # 增强500张数据集
└── c/                    # 增强1000张数据集
```

## 使用步骤

### 1. 环境准备

```bash
# 安装环境
bash setup_yolov8.sh

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./yolov8_env
```

### 2. 数据准备

```bash
# 转换数据为YOLO格式
python prepare_data.py
```

### 3. 训练与评估

```bash
# 开始训练和评估
python train_eval.py
```

### 4. 查看结果

训练完成后，结果将保存在`results`目录：

- `performance_comparison.png`：可视化性能对比
- `summary_report.txt`：详细评估报告

## 配置说明

- 在`train_eval.py`中可调整：
  - 训练轮数（epochs）
  - 图像大小（imgsz）
  - 批次大小（batch）
  - GPU设备（device）

## 数据集说明

- `original`：原始数据集
- `a`：增强100张图像的数据集
- `b`：增强500张图像的数据集
- `c`：增强1000张图像的数据集 