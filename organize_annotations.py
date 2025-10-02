import os
import shutil
from pathlib import Path
import glob

# 路径设置
BASE_DIR = "/research/projects/trans_llm/Xiang_Zhang/In-Vitro-Machine-Learning-Based-CAR-T-Immunological-Synapse-Quality"
TRAIN_DIR = f"{BASE_DIR}/data/train"
VAL_DIR = f"{BASE_DIR}/data/val"
TEST_DIR = f"{BASE_DIR}/data/test"
MASKS_DIR = f"{BASE_DIR}/data/masks"

# 创建annotation目录
TRAIN_ANNO_DIR = f"{BASE_DIR}/train_annotation"
VAL_ANNO_DIR = f"{BASE_DIR}/val_annotation"
TEST_ANNO_DIR = f"{BASE_DIR}/test_annotation"

os.makedirs(TRAIN_ANNO_DIR, exist_ok=True)
os.makedirs(VAL_ANNO_DIR, exist_ok=True)
os.makedirs(TEST_ANNO_DIR, exist_ok=True)

def process_dataset(src_dir, dest_dir, dataset_name):
    """处理单个数据集目录，查找对应的masks"""
    print(f"处理{dataset_name}数据集...")
    
    # 获取所有图像文件
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(src_dir, ext)))
    
    total_images = len(image_files)
    found_masks = 0
    
    for img_path in image_files:
        # 提取基本文件名（无扩展名）
        base_name = Path(img_path).stem
        
        # 查找对应的mask文件
        mask_path = os.path.join(MASKS_DIR, f"{base_name}.png")
        
        # 如果mask存在，复制到目标目录
        if os.path.exists(mask_path):
            shutil.copy(mask_path, os.path.join(dest_dir, f"{base_name}.png"))
            found_masks += 1
        else:
            print(f"警告: 未找到图像 {base_name} 的mask")
    
    print(f"{dataset_name}数据集处理完成: 共{total_images}个图像，成功找到{found_masks}个对应mask")
    return total_images, found_masks

# 处理各个数据集
train_total, train_found = process_dataset(TRAIN_DIR, TRAIN_ANNO_DIR, "训练")
val_total, val_found = process_dataset(VAL_DIR, VAL_ANNO_DIR, "验证")
test_total, test_found = process_dataset(TEST_DIR, TEST_ANNO_DIR, "测试")

# 打印总结
print("\n===== 处理摘要 =====")
print(f"训练集: {train_found}/{train_total} 找到对应mask，成功率: {train_found/train_total*100:.1f}%")
print(f"验证集: {val_found}/{val_total} 找到对应mask，成功率: {val_found/val_total*100:.1f}%")
print(f"测试集: {test_found}/{test_total} 找到对应mask，成功率: {test_found/test_total*100:.1f}%")
print(f"总计: {train_found+val_found+test_found}/{train_total+val_total+test_total} 找到对应mask")
