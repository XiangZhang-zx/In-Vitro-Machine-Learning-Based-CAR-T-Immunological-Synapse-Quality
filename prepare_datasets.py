import os
import shutil
from pathlib import Path

def process_image_files(group="A"):
    """
    将_fake文件复制到train目录，_real文件复制到train_annotation目录
    并去掉_fake和_real后缀
    """
    # 源目录和目标目录
    src_dir = f"/research/projects/trans_llm/Xiang_Zhang/data/ControlNet/training/cell_dataset/pytorch-CycleGAN-and-pix2pix/large_images/large_images_{group}"
    target_train_dir = f"/research/projects/trans_llm/Xiang_Zhang/In-Vitro-Machine-Learning-Based-CAR-T-Immunological-Synapse-Quality/dataset/{group.lower()}/train"
    target_anno_dir = f"/research/projects/trans_llm/Xiang_Zhang/In-Vitro-Machine-Learning-Based-CAR-T-Immunological-Synapse-Quality/dataset/{group.lower()}/train_annotation"
    
    # 确保目标目录存在
    os.makedirs(target_train_dir, exist_ok=True)
    os.makedirs(target_anno_dir, exist_ok=True)
    
    # 获取所有图像文件
    files = os.listdir(src_dir)
    
    # 计数器
    fake_count = 0
    real_count = 0
    
    for file in files:
        if not file.endswith('.png'):
            continue
            
        src_path = os.path.join(src_dir, file)
        
        # 处理_fake文件
        if '_fake.png' in file:
            # 去掉_fake后缀
            new_name = file.replace('_fake.png', '.png')
            target_path = os.path.join(target_train_dir, new_name)
            
            # 复制文件
            shutil.copy2(src_path, target_path)
            fake_count += 1
            
        # 处理_real文件
        elif '_real.png' in file:
            # 去掉_real后缀
            new_name = file.replace('_real.png', '.png')
            target_path = os.path.join(target_anno_dir, new_name)
            
            # 复制文件
            shutil.copy2(src_path, target_path)
            real_count += 1
    
    print(f"处理完成！")
    print(f"复制了 {fake_count} 个_fake图像到 {target_train_dir}")
    print(f"复制了 {real_count} 个_real图像到 {target_anno_dir}")

def process_all_groups():
    """处理A、B、C三组数据"""
    for group in ["A", "B", "C"]:
        print(f"\n开始处理{group}组数据...")
        process_image_files(group)

if __name__ == "__main__":
    # 处理A组数据
    # process_image_files("A")
    
    # 如果需要处理所有组，取消下面的注释
    process_all_groups()

