#!/usr/bin/env python3
"""
Fine-tune Cellpose model on your own data
"""

import os
import argparse
import numpy as np
from pathlib import Path
from cellpose import models, io, train
from cellpose.io import logger_setup
import logging
import json
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def prepare_training_data(image_dir, mask_dir, output_dir, max_images=None):
    """
    Prepare training data by pairing images with masks
    
    Args:
        image_dir: Directory with original images
        mask_dir: Directory with mask files (*_cp_masks.png)
        output_dir: Directory to save training data
        max_images: Maximum number of images to use (None for all)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in ['*.png', '*.tif', '*.tiff', '*.jpg', '*.jpeg']:
        image_files.extend(Path(image_dir).glob(ext))
    
    # Get all mask files
    mask_files = list(Path(mask_dir).glob('*_cp_masks.png'))
    
    print(f"Found {len(image_files)} images and {len(mask_files)} masks")
    
    # Match images with masks
    paired_data = []
    for img_file in image_files:
        # Try to find corresponding mask
        img_stem = img_file.stem
        
        # Look for mask with similar name
        for mask_file in mask_files:
            mask_stem = mask_file.stem.replace('_cp_masks', '')
            if img_stem == mask_stem or img_stem in mask_stem or mask_stem in img_stem:
                paired_data.append((img_file, mask_file))
                break
    
    print(f"Successfully paired {len(paired_data)} image-mask pairs")
    
    if max_images:
        paired_data = paired_data[:max_images]
        print(f"Using first {len(paired_data)} pairs for training")
    
    # Copy and prepare data
    train_images = []
    train_masks = []
    
    for i, (img_file, mask_file) in enumerate(paired_data):
        # Load image
        img = io.imread(str(img_file))
        
        # Load mask
        mask = io.imread(str(mask_file))
        
        # Save to training directory
        train_img_path = os.path.join(output_dir, f"train_img_{i:04d}.png")
        train_mask_path = os.path.join(output_dir, f"train_img_{i:04d}_seg.npy")
        
        io.imsave(train_img_path, img)
        np.save(train_mask_path, mask)
        
        train_images.append(train_img_path)
        train_masks.append(train_mask_path)
    
    return train_images, train_masks

def finetune_cellpose(train_dir, model_name="cyto", custom_model_name="my_cellpose_model", 
                     n_epochs=100, learning_rate=0.1, batch_size=8):
    """
    Fine-tune Cellpose model
    
    Args:
        train_dir: Directory with training data
        model_name: Base model to start from ('cyto', 'nuclei', 'cyto2', etc.)
        custom_model_name: Name for your custom model
        n_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
    """
    
    # Setup logging
    logger_setup()
    
    # Get training files
    train_files = []
    for f in os.listdir(train_dir):
        if f.endswith('.png') and not f.endswith('_seg.png'):
            train_files.append(os.path.join(train_dir, f))
    
    print(f"Found {len(train_files)} training images")
    
    # Load training data using Cellpose's data loader
    print(f"Loading training data...")
    output = io.load_train_test_data(train_dir, mask_filter='_masks')
    train_data, train_labels, image_names, test_data, test_labels, test_names = output

    print(f"Loaded {len(train_data)} training images")
    print(f"Loaded {len(train_labels)} training masks")

    # Load base model
    print(f"Loading base model: {model_name}")
    model = models.CellposeModel(model_type=model_name, gpu=True)

    # Train the model
    print(f"Starting fine-tuning...")
    print(f"Model name: {custom_model_name}")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")

    # Use train module for Cellpose 4.0+
    new_model_path = train.train_seg(
        model.net,
        train_data=train_data,
        train_labels=train_labels,
        save_path=train_dir,
        model_name=custom_model_name,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=0.0001,
        batch_size=batch_size,
        min_train_masks=1
    )

    print(f"Training completed! Model saved to: {new_model_path}")
    return new_model_path


def mask_to_bbox(mask):
    """
    Convert a binary mask to bounding box [x, y, width, height]
    """
    pos = np.where(mask)
    if len(pos[0]) == 0:
        return None

    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])

    width = xmax - xmin + 1
    height = ymax - ymin + 1

    return [int(xmin), int(ymin), int(width), int(height)]


def evaluate_on_coco(model_path, val_ann_file, val_image_dir):
    """
    Evaluate Cellpose model on COCO format validation set

    Args:
        model_path: Path to trained Cellpose model
        val_ann_file: Path to COCO format validation annotations
        val_image_dir: Directory containing validation images

    Returns:
        dict: Evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"📊 评估Cellpose模型 - COCO格式")
    print(f"{'='*60}")
    print(f"模型: {model_path}")
    print(f"验证集: {val_ann_file}")

    # Load COCO ground truth
    coco_gt = COCO(val_ann_file)

    # Load model
    print("\n🤖 加载Cellpose模型...")
    model = models.CellposeModel(gpu=True, pretrained_model=model_path)

    # Prepare results
    results = []

    # Get all images
    image_ids = coco_gt.getImgIds()
    print(f"\n🔍 处理 {len(image_ids)} 张验证图片...")

    for idx, img_id in enumerate(image_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(val_image_dir, img_info['file_name'])

        if not os.path.exists(img_path):
            print(f"⚠️  图片不存在: {img_path}")
            continue

        # Load and predict
        img = io.imread(img_path)
        masks, flows, styles = model.eval(img, diameter=None, channels=[0, 0])

        # Convert masks to COCO format
        unique_ids = np.unique(masks)
        unique_ids = unique_ids[unique_ids > 0]  # Remove background

        for cell_id in unique_ids:
            cell_mask = (masks == cell_id).astype(np.uint8)
            bbox = mask_to_bbox(cell_mask)

            if bbox is None:
                continue

            # Calculate area
            area = int(np.sum(cell_mask))

            # Add to results
            results.append({
                'image_id': img_id,
                'category_id': 1,  # cell
                'bbox': bbox,
                'area': area,
                'score': 1.0,  # Cellpose doesn't provide confidence scores
                'segmentation': []  # We only use bbox for evaluation
            })

        if (idx + 1) % 10 == 0:
            print(f"   处理进度: {idx + 1}/{len(image_ids)}")

    print(f"\n✅ 检测完成！共检测到 {len(results)} 个细胞")

    # Save results to temporary file
    results_file = 'cellpose_val_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f)

    print(f"💾 结果已保存到: {results_file}")

    # Evaluate using COCO API
    print(f"\n{'='*60}")
    print(f"📈 计算COCO指标...")
    print(f"{'='*60}")

    coco_dt = coco_gt.loadRes(results_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract metrics
    metrics = {
        'mAP': coco_eval.stats[0],
        'mAP@50': coco_eval.stats[1],
        'mAP@75': coco_eval.stats[2],
        'mAP_small': coco_eval.stats[3],
        'mAP_medium': coco_eval.stats[4],
        'mAP_large': coco_eval.stats[5],
        'AR@1': coco_eval.stats[6],
        'AR@10': coco_eval.stats[7],
        'AR@100': coco_eval.stats[8],
    }

    # Save summary
    summary_file = 'cellpose_val_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n💾 评估结果已保存到: {summary_file}")

    return metrics


def test_model(model_path, test_image_path, output_dir):
    """
    Test the fine-tuned model on a sample image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load custom model
    model = models.CellposeModel(pretrained_model=model_path, gpu=True)
    
    # Load test image
    img = io.imread(test_image_path)
    
    # Run segmentation
    masks, flows, styles = model.eval(img, diameter=None, channels=[0, 0])
    
    # Save results
    base_name = Path(test_image_path).stem
    io.save_masks(img, masks, flows, test_image_path, 
                  png=True, save_dir=output_dir, suffix=f"_{base_name}_custom")
    
    print(f"Test results saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Cellpose model")
    parser.add_argument("--augmented", action="store_true",
                       help="使用增强数据集 (193张) 而不是原始数据集 (93张)")
    parser.add_argument("--mode", choices=["prepare", "train", "test", "full"],
                       default="train", help="Mode to run")
    parser.add_argument("--model_name", default="cyto2",
                       help="Base model to start from")
    parser.add_argument("--max_images", type=int, default=None,
                       help="Max images for training (None for all)")
    parser.add_argument("--n_epochs", type=int, default=200,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.1,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--test_image", help="Path to test image")
    parser.add_argument("--test_output", default="cellpose_test_results",
                       help="Directory for test results")

    args = parser.parse_args()

    # 根据参数选择数据集路径
    if args.augmented:
        base_path = "In-Vitro-Machine-Learning-Based-CAR-T-Immunological-Synapse-Quality/augument_dataset"
        custom_model_name = "cellpose_cell_model_augmented"
        summary_file = "cellpose_val_summary_augmented.json"
        dataset_name = "增强数据集 (193张)"
    else:
        base_path = "In-Vitro-Machine-Learning-Based-CAR-T-Immunological-Synapse-Quality/dataset"
        custom_model_name = "cellpose_cell_model_fixed"
        summary_file = "cellpose_val_summary.json"
        dataset_name = "原始数据集 (93张)"

    # 设置路径
    train_dir = f"{base_path}/cellpose_format_fixed/train"
    val_ann_file = f"{base_path}/coco_format_fixed/val_annotations.json"
    val_image_dir = f"{base_path}/coco_format_fixed/val"

    print("=" * 60)
    print(f"📊 数据集: {dataset_name}")
    print(f"📁 训练目录: {train_dir}")
    print("=" * 60)

    if args.mode in ["prepare", "full"]:
        print("=== Preparing Training Data ===")
        train_images, train_masks = prepare_training_data(
            train_dir, train_dir, train_dir, args.max_images
        )

    if args.mode in ["train", "full"]:
        print("=== Fine-tuning Model ===")
        model_path = finetune_cellpose(
            train_dir, args.model_name, custom_model_name,
            args.n_epochs, args.learning_rate, args.batch_size
        )

        # Automatically evaluate after training
        print("\n" + "="*60)
        print("🎯 训练完成！开始在验证集上评估...")
        print("="*60)

        # Find the trained model
        model_files = list(Path(train_dir).glob(f"models/{custom_model_name}*"))
        if not model_files:
            # Try without models/ subdirectory
            model_files = list(Path(train_dir).glob(f"{custom_model_name}*"))

        if model_files:
            trained_model_path = str(model_files[0])
            print(f"✅ 找到训练好的模型: {trained_model_path}")

            # Evaluate on validation set
            metrics = evaluate_on_coco(
                trained_model_path,
                val_ann_file,
                val_image_dir
            )

            # Save with correct filename
            with open(summary_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"\n{'='*60}")
            print(f"🏆 最终评估结果 ({dataset_name}):")
            print(f"{'='*60}")
            print(f"   mAP@50-95: {metrics['mAP']:.4f} ({metrics['mAP']*100:.2f}%)")
            print(f"   mAP@50:    {metrics['mAP@50']:.4f} ({metrics['mAP@50']*100:.2f}%)")
            print(f"   mAP@75:    {metrics['mAP@75']:.4f} ({metrics['mAP@75']*100:.2f}%)")
            print(f"💾 结果已保存到: {summary_file}")
            print(f"{'='*60}")
        else:
            print("⚠️  未找到训练好的模型，跳过评估")

    if args.mode == "test" and args.test_image:
        print("=== Testing Model ===")
        # Find the model file
        model_files = list(Path(train_dir).glob(f"{custom_model_name}*"))
        if model_files:
            test_model(str(model_files[0]), args.test_image, args.test_output)
        else:
            print("No trained model found!")

if __name__ == "__main__":
    main()
