from dataset import TestDataset
from unet_model import Unet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import numpy as np
import os
import argparse

# 类别定义和颜色映射
CLASS_NAMES = ['background', 'soil', 'water', 'road', 'field']
COLOR_MAP = {
    0: (0, 0, 0),       # 背景
    1: (255, 215, 0),   # soil 金色
    2: (255, 0, 0),     # water 红色
    3: (0, 255, 0),     # road 绿色
    4: (0, 0, 255),     # field 蓝色
}

def label_to_color(label_array):
    """将单通道类别索引转换为 RGB 彩色图像"""
    h, w = label_array.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        rgb_img[label_array == class_id] = color
    return rgb_img

def test_model(model_path, images_dir, output_dir):
    """
    测试模型并保存预测结果
    """
    # 检查目录是否存在
    if not os.path.exists(images_dir):
        print(f"错误：图像目录不存在：{images_dir}")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    test_dataset = TestDataset(images_dir=images_dir, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # 加载模型 - 支持 checkpoint 和 state_dict 两种格式
    model = Unet(3, 5)  # 5 类别
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # checkpoint 格式（包含 optimizer 等）
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载 checkpoint: {model_path} (epoch {checkpoint.get('epoch', '?')})")
    else:
        # state_dict 格式（旧版本）
        model.load_state_dict(checkpoint)
        print(f"模型已加载：{model_path}")
    model = model.to(device)
    model = model.eval()

    with torch.no_grad():
        for image, path in tqdm(test_dataloader, desc="预测中"):
            image = image.to(device)
            output = model(image)
            # 多分类：使用 argmax 获取类别
            output = torch.argmax(output, dim=1)

            # 将预测结果转为 RGB 彩色图像保存
            output_np = output.squeeze().cpu().numpy()
            rgb_img = label_to_color(output_np)
            output_img = Image.fromarray(rgb_img)

            base_name = os.path.basename(path[0])
            name = os.path.splitext(base_name)[0] + '_pred.png'
            output_img.save(os.path.join(output_dir, name))

    print(f"\n测试完成，预测结果保存在：{output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='测试四分类分割模型')
    parser.add_argument('--model', type=str,
                        default=r'./model/model_epoch_.pth',
                        help='模型权重路径')
    parser.add_argument('--images', type=str,
                        default=r'D:\BaiduNetdiskDownload\农田数据集-UJN_Land\biaozhu\biaozhu\shandong\suanfa_sample-1000\test',
                        help='测试图像目录')
    parser.add_argument('--output', type=str,
                        default=r'predictions',
                        help='输出目录')

    args = parser.parse_args()

    test_model(
        model_path=args.model,
        images_dir=args.images,
        output_dir=args.output
    )
