import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# 简单数据集类
class CellDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # 读取图像和掩码
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        
        # 将RGB掩码转换为二值掩码（有细胞=1，背景=0）
        binary_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        # 如果任何RGB通道有值，则认为是细胞
        binary_mask[np.any(mask > 0, axis=2)] = 1
        
        # 调整图像大小为256x256以加速训练
        image = Image.fromarray(image).resize((256, 256))
        binary_mask = Image.fromarray(binary_mask).resize((256, 256), Image.NEAREST)
        
        # 转换为numpy数组
        image = np.array(image) / 255.0  # 归一化到[0,1]
        binary_mask = np.array(binary_mask)
        
        # 转换为PyTorch张量
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        binary_mask = torch.from_numpy(binary_mask).unsqueeze(0).float()
        
        return image, binary_mask

# 简单的U-Net模型
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        # 编码器部分
        self.enc1 = self._double_conv(3, 64)
        self.enc2 = self._double_conv(64, 128)
        self.enc3 = self._double_conv(128, 256)
        
        # 解码器部分
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._double_conv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._double_conv(128, 64)
        
        # 输出层
        self.outconv = nn.Conv2d(64, 1, kernel_size=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2)
    
    def _double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 编码路径
        e1 = self.enc1(x)
        e1_pool = self.pool(e1)
        
        e2 = self.enc2(e1_pool)
        e2_pool = self.pool(e2)
        
        e3 = self.enc3(e2_pool)
        
        # 解码路径（带跳跃连接）
        d2 = self.upconv2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.outconv(d1)
        return torch.sigmoid(out)

# 训练函数
def train_model(model, train_loader, val_loader, device, num_epochs=5):
    # 二值交叉熵损失函数
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"训练 轮次{epoch+1}/{num_epochs}"):
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"验证 轮次{epoch+1}/{num_epochs}"):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # 打印指标
        print(f"轮次 {epoch+1}/{num_epochs}")
        print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("保存最佳模型...")
            model_path = os.path.join(".", "best_model.pth")
            torch.save(model.state_dict(), model_path)
    
    return model

# 计算IOU和Dice性能指标
def calculate_metrics(model, data_loader, device):
    model.eval()
    iou_scores = []
    dice_scores = []
    
    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc="计算性能指标"):
            images = images.to(device)
            true_masks = masks.to(device)
            
            # 获取预测
            outputs = model(images)
            pred_masks = (outputs > 0.5).float()
            
            batch_size = true_masks.size(0)
            
            for i in range(batch_size):
                pred = pred_masks[i].cpu().numpy()
                true = true_masks[i].cpu().numpy()
                
                # 计算IoU
                pred_flat = pred.flatten()
                true_flat = true.flatten()
                
                intersection = np.logical_and(pred_flat, true_flat).sum()
                union = np.logical_or(pred_flat, true_flat).sum()
                
                iou = intersection / (union + 1e-6)
                iou_scores.append(iou)
                
                # 计算Dice系数
                dice = (2.0 * intersection) / (pred_flat.sum() + true_flat.sum() + 1e-6)
                dice_scores.append(dice)
    
    # 计算平均IoU和Dice
    avg_iou = np.mean(iou_scores)
    avg_dice = np.mean(dice_scores)
    
    return {
        "平均IoU": avg_iou,
        "平均Dice系数": avg_dice
    }

# 实验设置
def main():
    # 数据目录
    base_dir = '/research/projects/trans_llm/Xiang_Zhang/In-Vitro-Machine-Learning-Based-CAR-T-Immunological-Synapse-Quality/dataset/original'
    train_img_dir = os.path.join(base_dir, 'train')
    train_mask_dir = os.path.join(base_dir, 'train_annotation')
    val_img_dir = os.path.join(base_dir, 'val')
    val_mask_dir = os.path.join(base_dir, 'val_annotation')
    test_img_dir = os.path.join(base_dir, 'test')
    test_mask_dir = os.path.join(base_dir, 'test_annotation')
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据集
    train_dataset = CellDataset(train_img_dir, train_mask_dir)
    val_dataset = CellDataset(val_img_dir, val_mask_dir)
    test_dataset = CellDataset(test_img_dir, test_mask_dir)
    
    # 数据加载器
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建模型
    model = SimpleUNet().to(device)
    
    # 训练模型
    print("开始训练...")
    model = train_model(model, train_loader, val_loader, device, num_epochs=5)
    
    # 评估模型
    print("模型评估...")
    metrics = calculate_metrics(model, test_loader, device)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    print("训练和评估完成！")

if __name__ == "__main__":
    main()
