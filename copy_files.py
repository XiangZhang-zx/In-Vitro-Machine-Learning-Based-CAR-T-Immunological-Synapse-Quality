#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

# 源目录和目标目录
source_dir = "/research/projects/trans_llm/Xiang_Zhang/In-Vitro-Machine-Learning-Based-CAR-T-Immunological-Synapse-Quality/dataset/original"
dest_dirs = [
    "/research/projects/trans_llm/Xiang_Zhang/In-Vitro-Machine-Learning-Based-CAR-T-Immunological-Synapse-Quality/dataset/a",
    "/research/projects/trans_llm/Xiang_Zhang/In-Vitro-Machine-Learning-Based-CAR-T-Immunological-Synapse-Quality/dataset/b",
    "/research/projects/trans_llm/Xiang_Zhang/In-Vitro-Machine-Learning-Based-CAR-T-Immunological-Synapse-Quality/dataset/c"
]

def main():
    # 检查源目录是否存在
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"错误：源目录 {source_dir} 不存在")
        return
    
    # 创建目标目录（如果不存在）
    for dest_dir in dest_dirs:
        dest_path = Path(dest_dir)
        if not dest_path.exists():
            print(f"创建目录：{dest_dir}")
            os.makedirs(dest_path, exist_ok=True)
    
    # 获取源目录中的所有子目录
    subdirs = [d for d in source_path.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"警告：源目录 {source_dir} 中没有子目录")
        return
    
    print(f"找到 {len(subdirs)} 个子目录需要复制")
    
    # 复制整个目录结构到每个目标目录
    for dest_dir in dest_dirs:
        print(f"正在复制目录结构到 {dest_dir}...")
        for subdir in subdirs:
            dest_subdir = Path(dest_dir) / subdir.name
            if dest_subdir.exists():
                print(f"  目标子目录已存在: {dest_subdir}，跳过创建")
            else:
                print(f"  创建子目录: {dest_subdir}")
                shutil.copytree(subdir, dest_subdir)
                print(f"  已复制: {subdir.name} 及其所有内容")
    
    print("所有目录和文件复制完成！")

if __name__ == "__main__":
    main()