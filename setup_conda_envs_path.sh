#!/bin/bash

# 设置自定义的 conda 环境路径
# 类似于 Zeru_Shi 的做法

# 创建环境目录
mkdir -p ./conda_envs

# 设置 conda 环境路径
export CONDA_ENVS_PATH="$(pwd)/conda_envs:/research/projects/trans_llm/Xiang_Zhang/cellpose/miniconda3/envs"

# 添加到 conda 配置
conda config --add envs_dirs $(pwd)/conda_envs

echo "✅ 设置完成！"
echo "📍 自定义环境路径：$(pwd)/conda_envs"
echo ""
echo "现在您可以直接使用 conda create -n myenv 创建环境"
echo "环境将自动存储在本地 conda_envs 目录中"
echo ""
echo "查看所有环境路径："
conda config --show envs_dirs
