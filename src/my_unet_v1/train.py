import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from unet_model import Unet
from dataset import SegmentationDataset
import tqdm
import argparse
import os

# 配置
transform = transforms.Compose([
    transforms.ToTensor(),  # 把图像转换为一个张量，同时值变为 0-1
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
])

mask_transform = None  # 不使用 ToTensor 归一化，直接在 dataset 中转为 tensor

train_dataset = SegmentationDataset(
    images_dir=r'D:\BaiduNetdiskDownload\农田数据集-UJN_Land\biaozhu\biaozhu\shandong\suanfa_sample-1000\jpg',
    masks_dir=r'D:\BaiduNetdiskDownload\农田数据集-UJN_Land\biaozhu\biaozhu\shandong\suanfa_sample-1000\png_single',
    transform=transform,
    mask_transform=mask_transform
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(resume=False, start_epoch=0, epochs=50, save_interval=10, model_dir='./model'):
    """
    训练函数

    Args:
        resume: 是否从之前的checkpoint继续训练
        start_epoch: 起始epoch（从 0 开始，resume时设置）
        epochs: 总epoch数
        save_interval: 每多少epoch保存一次模型
        model_dir: 模型保存目录
    """
    os.makedirs(model_dir, exist_ok=True)

    model = Unet(3, 5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 如果 resume 为 True，加载最新的 checkpoint
    if resume:
        # 查找最新的模型文件
        model_files = [f for f in os.listdir(model_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
        if model_files:
            # 按 epoch 排序
            model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            latest_model = model_files[-1]
            if latest_model != 'model_epoch_.pth':  # 排除最终输出文件
                checkpoint_path = os.path.join(model_dir, latest_model)
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"从 checkpoint 恢复：{checkpoint_path} (epoch {checkpoint['epoch']})")
            else:
                print("未找到可恢复的 checkpoint，从头开始训练")
                start_epoch = 0
        else:
            print("未找到模型文件，从头开始训练")
            start_epoch = 0

    print(f"使用设备：{device}")
    print(f"训练轮数：{start_epoch} -> {epochs}")
    print(f"保存间隔：每 {save_interval} 个 epoch")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0

        for images, masks in tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # 每 save_interval 个 epoch 保存一次
        if (epoch + 1) % save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            save_path = os.path.join(model_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(checkpoint, save_path)
            print(f"模型已保存：{save_path}")

    # 保存最终模型
    final_path = os.path.join(model_dir, 'model_epoch_.pth')
    torch.save(model.state_dict(), final_path)
    print(f"最终模型已保存：{final_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练四分类分割模型')
    parser.add_argument('--resume', action='store_true', help='是否从 checkpoint 继续训练')
    parser.add_argument('--epochs', type=int, default=50, help='总训练轮数')
    parser.add_argument('--start-epoch', type=int, default=0, help='起始 epoch')
    parser.add_argument('--save-interval', type=int, default=10, help='每多少 epoch 保存一次')
    parser.add_argument('--model-dir', type=str, default='./model', help='模型保存目录')

    args = parser.parse_args()

    train(
        resume=args.resume,
        start_epoch=args.start_epoch,
        epochs=args.epochs,
        save_interval=args.save_interval,
        model_dir=args.model_dir
    )