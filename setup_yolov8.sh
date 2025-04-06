#!/bin/bash

# 创建YOLOv8环境
echo "===== 创建YOLOv8环境 ====="
conda create --prefix ./yolov8_env python=3.9 -y

# 激活环境
echo "===== 激活环境 ====="
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./yolov8_env

# 安装必要包
echo "===== 安装必要包 ====="
pip install ultralytics
pip install opencv-python matplotlib

# 下载预训练模型
echo "===== 下载预训练模型 ====="
mkdir -p pretrained_models
cd pretrained_models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt
cd ..

echo "===== 安装完成 ====="
echo "使用以下命令激活环境："
echo "source $(conda info --base)/etc/profile.d/conda.sh"
echo "conda activate ./yolov8_env"
echo ""
echo "然后运行训练脚本："
echo "python train_eval.py" 