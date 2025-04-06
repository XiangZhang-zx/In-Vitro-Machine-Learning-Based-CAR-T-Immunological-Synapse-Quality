import os
import cv2
import numpy as np
import glob
import shutil
from tqdm import tqdm

def mask_to_yolo_segmentation(mask_path, img_shape):
    """将掩码图像转换为YOLO格式的分割标注"""
    # 读取掩码
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"警告：无法读取掩码 {mask_path}")
        return ""
    
    # 二值化（如果需要）
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return ""
    
    # 对于每个轮廓，生成YOLO格式的标注
    yolo_annotations = []
    h, w = img_shape[:2]
    
    for contour in contours:
        # 简化轮廓以减少点数
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 如果轮廓太小，跳过
        if len(approx) < 3 or cv2.contourArea(approx) < 10:
            continue
        
        # 转换为YOLO格式（归一化坐标）
        points = []
        for point in approx:
            x, y = point[0]
            points.append(f"{x/w} {y/h}")
        
        # 添加类别和点坐标
        yolo_annotations.append(f"0 {' '.join(points)}")
    
    return "\n".join(yolo_annotations)

def convert_dataset(source_dir, target_dir):
    """转换数据集到YOLO格式"""
    # 创建目录结构
    os.makedirs(os.path.join(target_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "labels"), exist_ok=True)
    
    # 获取所有图像文件
    image_files = glob.glob(os.path.join(source_dir, "images", "*.jpg")) + \
                 glob.glob(os.path.join(source_dir, "images", "*.png"))
    
    print(f"处理 {len(image_files)} 图像...")
    
    for img_path in tqdm(image_files):
        # 获取文件名和扩展名
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        
        # 读取图像获取尺寸
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取图像 {img_path}")
            continue
        
        # 查找对应的掩码文件
        mask_path = os.path.join(source_dir, "masks", f"{base_name}.png")
        if not os.path.exists(mask_path):
            print(f"警告：找不到掩码 {mask_path}")
            continue
        
        # 转换掩码为YOLO格式
        yolo_annotation = mask_to_yolo_segmentation(mask_path, img.shape)
        if not yolo_annotation:
            print(f"警告：无法为 {img_name} 创建有效的标注")
            continue
        
        # 保存图像
        shutil.copy(img_path, os.path.join(target_dir, "images", img_name))
        
        # 保存标注
        with open(os.path.join(target_dir, "labels", f"{base_name}.txt"), "w") as f:
            f.write(yolo_annotation)
    
    print(f"转换完成！数据保存到 {target_dir}")

def process_all_datasets():
    """处理所有数据集"""
    datasets = ['original', 'a', 'b', 'c']
    phases = ['train', 'val', 'test']
    
    for dataset in datasets:
        print(f"\n===== 处理数据集 {dataset} =====")
        for phase in phases:
            source = f"./{dataset}/{phase}"
            target = f"./{dataset}_yolo/{phase}"
            print(f"转换 {source} 到 {target}")
            convert_dataset(source, target)
        
        # 更新YAML文件指向新的YOLO格式数据
        with open(f"data_{dataset}.yaml", "r") as f:
            yaml_content = f.read()
        
        yaml_content = yaml_content.replace(f"path: ./{dataset}", f"path: ./{dataset}_yolo")
        
        with open(f"data_{dataset}.yaml", "w") as f:
            f.write(yaml_content)

if __name__ == "__main__":
    print("开始准备YOLO格式数据...")
    process_all_datasets()
    print("\n所有数据集准备完成！") 