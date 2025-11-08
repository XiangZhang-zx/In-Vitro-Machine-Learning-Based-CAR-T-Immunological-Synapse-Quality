#!/usr/bin/env python
"""
MMDetection训练脚本 - 细胞检测
支持4个数据集:
1. cart_original: CAR-T原始数据集 (93张)
2. cart_augmented: CAR-T增强数据集 (193张)
3. kaggle_original: Kaggle原始数据集 (669张)
4. kaggle_augmented: Kaggle增强数据集 (862张)
"""

import os
import sys
import argparse

# 添加mmdetection到路径
sys.path.insert(0, 'mmdetection')

from mmengine.config import Config
from mmengine.runner import Runner

def main():
    parser = argparse.ArgumentParser(description='训练Faster R-CNN细胞检测模型')
    parser.add_argument('--dataset', type=str,
                       choices=['cart_original', 'cart_augmented', 'kaggle_original', 'kaggle_augmented'],
                       default='cart_original',
                       help='数据集选择: cart_original (93), cart_augmented (193), kaggle_original (669), kaggle_augmented (862)')
    args = parser.parse_args()

    # 根据参数选择配置文件和数据集名称
    dataset_configs = {
        'cart_original': {
            'config': 'configs/faster_rcnn_cell_detection_cart_original.py',
            'name': 'CAR-T原始数据集 (93张)',
            'images': 93
        },
        'cart_augmented': {
            'config': 'configs/faster_rcnn_cell_detection_cart_augmented.py',
            'name': 'CAR-T增强数据集 (193张)',
            'images': 193
        },
        'kaggle_original': {
            'config': 'configs/faster_rcnn_cell_detection_kaggle_original.py',
            'name': 'Kaggle原始数据集 (669张)',
            'images': 669
        },
        'kaggle_augmented': {
            'config': 'configs/faster_rcnn_cell_detection_kaggle_augmented.py',
            'name': 'Kaggle增强数据集 (862张)',
            'images': 862
        }
    }

    config_info = dataset_configs[args.dataset]
    config_file = config_info['config']
    dataset_name = config_info['name']

    cfg = Config.fromfile(config_file)

    # 创建工作目录
    os.makedirs(cfg.work_dir, exist_ok=True)

    print("=" * 80)
    print("🚀 开始训练 Faster R-CNN 细胞检测模型")
    print("=" * 80)
    print(f"📁 配置文件: {config_file}")
    print(f"📊 数据集: {dataset_name}")
    print(f"📁 工作目录: {cfg.work_dir}")
    print(f"📊 训练轮数: {cfg.train_cfg.max_epochs}")
    print(f"📊 批次大小: {cfg.train_dataloader.batch_size}")
    print(f"📊 学习率: {cfg.optim_wrapper.optimizer.lr}")
    print(f"🔧 Backbone: ResNet-101")
    print("=" * 80)

    # 创建runner并开始训练
    runner = Runner.from_cfg(cfg)
    runner.train()

    print("\n" + "=" * 80)
    print("✅ 训练完成！")
    print("=" * 80)
    print(f"📊 数据集: {dataset_name}")
    print(f"📁 结果保存在: {cfg.work_dir}")
    print("=" * 80)

if __name__ == '__main__':
    main()

