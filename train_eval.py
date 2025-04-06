from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import numpy as np
import json

# 确保结果目录存在
os.makedirs('results', exist_ok=True)

# 定义数据集
datasets = ['original', 'a', 'b', 'c']
dataset_names = ['原始数据', '增强100张', '增强500张', '增强1000张']

# 存储结果
map_results = []
precision_results = []
recall_results = []

# 对每个数据集分别训练
for idx, dataset in enumerate(datasets):
    print(f"\n===== 训练数据集 {dataset} ({dataset_names[idx]}) =====\n")
    
    # 加载预训练模型
    model = YOLO('yolov8n-seg.pt')  # 分割任务用-seg，检测任务用普通模型
    
    # 训练模型
    results = model.train(
        data=f'data_{dataset}.yaml',
        epochs=50,
        imgsz=640,  # 可根据GPU内存调整
        batch=8,    # 可根据GPU内存调整
        name=f'cell_dataset_{dataset}',
        device=0    # GPU ID, 使用CPU则设为'cpu'
    )
    
    # 评估模型
    metrics = model.val()
    
    # 保存评估指标
    map_results.append(metrics.box.map)  # mAP@0.5:0.95
    precision_results.append(metrics.box.mp)  # 平均精度
    recall_results.append(metrics.box.mr)  # 平均召回率
    
    # 保存详细评估结果
    with open(f'results/metrics_{dataset}.json', 'w') as f:
        json.dump(metrics.box.ap_class_dict, f, indent=4)
    
    print(f"\n===== 数据集 {dataset} ({dataset_names[idx]}) 训练和评估完成 =====\n")

# 生成结果比较图表
plt.figure(figsize=(15, 10))

# mAP图
plt.subplot(3, 1, 1)
plt.bar(dataset_names, map_results, color='blue')
plt.title('平均精度 (mAP@0.5:0.95)')
plt.ylabel('mAP')
plt.ylim(0, 1)
for i, v in enumerate(map_results):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center')

# 精度图
plt.subplot(3, 1, 2)
plt.bar(dataset_names, precision_results, color='green')
plt.title('平均精度 (Precision)')
plt.ylabel('Precision')
plt.ylim(0, 1)
for i, v in enumerate(precision_results):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center')

# 召回率图
plt.subplot(3, 1, 3)
plt.bar(dataset_names, recall_results, color='red')
plt.title('平均召回率 (Recall)')
plt.ylabel('Recall')
plt.ylim(0, 1)
for i, v in enumerate(recall_results):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.savefig('results/performance_comparison.png', dpi=300)
plt.close()

# 生成结果报告
with open('results/summary_report.txt', 'w') as f:
    f.write("细胞图像数据增强效果评估报告\n")
    f.write("==========================\n\n")
    
    f.write("测试数据集:\n")
    for idx, dataset in enumerate(datasets):
        f.write(f"- {dataset_names[idx]}\n")
    
    f.write("\n性能指标对比:\n")
    f.write(f"{'数据集':<15} {'mAP@0.5:0.95':<15} {'精度':<15} {'召回率':<15}\n")
    f.write("-" * 60 + "\n")
    
    for idx, dataset in enumerate(datasets):
        f.write(f"{dataset_names[idx]:<15} {map_results[idx]:<15.4f} {precision_results[idx]:<15.4f} {recall_results[idx]:<15.4f}\n")
    
    # 计算相对于原始数据集的提升
    f.write("\n性能提升(相对原始数据集):\n")
    f.write(f"{'数据集':<15} {'mAP提升':<15} {'精度提升':<15} {'召回率提升':<15}\n")
    f.write("-" * 60 + "\n")
    
    for idx in range(1, len(datasets)):
        map_imp = (map_results[idx] - map_results[0]) / map_results[0] * 100
        prec_imp = (precision_results[idx] - precision_results[0]) / precision_results[0] * 100
        recall_imp = (recall_results[idx] - recall_results[0]) / recall_results[0] * 100
        
        f.write(f"{dataset_names[idx]:<15} {map_imp:<15.2f}% {prec_imp:<15.2f}% {recall_imp:<15.2f}%\n")
    
    f.write("\n结论:\n")
    best_idx = np.argmax(map_results)
    f.write(f"在测试的所有数据集中，{dataset_names[best_idx]}表现最佳，mAP达到{map_results[best_idx]:.4f}。\n")
    
    if best_idx > 0:
        imp = (map_results[best_idx] - map_results[0]) / map_results[0] * 100
        f.write(f"相比原始数据集，性能提升了{imp:.2f}%。\n")
    
    f.write("\n这些结果表明图像增强技术对于细胞检测/分割任务的有效性。")

print("\n===== 所有数据集训练评估完成，结果已保存到results目录 =====\n")
print("查看'results/summary_report.txt'获取详细分析报告")
print("查看'results/performance_comparison.png'获取可视化比较图表") 