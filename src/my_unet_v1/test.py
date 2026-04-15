from dataset import TestDataset
from unet_model import Unet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import numpy as nperr
import os
import argparse

def test_model(model_path, images_dir, output_dir, threshold=0.5):
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

    # 加载模型
    model = Unet(3, 1)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model = model.to(device)
    model = model.eval()
    print(f"模型已加载：{model_path}")

    with torch.no_grad():
        for image, path in tqdm(test_dataloader, desc="预测中"):
            image = image.to(device)
            output = model(image)
            output = torch.sigmoid(output)  # 对于二分类，用 sigmoid
            output = (output > threshold).float()

            # 将预测结果转为图像保存
            output_np = output.squeeze().cpu().numpy() * 255
            output_img = Image.fromarray(output_np.astype(np.uint8))

            base_name = os.path.basename(path[0])
            name = os.path.splitext(base_name)[0] + '_pred.png'
            output_img.save(os.path.join(output_dir, name))

    print(f"\n测试完成，预测结果保存在：{output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='测试小麦田分割模型')
    parser.add_argument('--model', type=str, default=r'D:\Develop\Remote-Sensing-Image-Wheat-Field-Segmentation\Remote-Sensing-Image-Wheat-Field-Segmentation\src\my_unet_v1\model\model_epoch_20.pth', help='模型权重路径')
    parser.add_argument('--images', type=str, default=r'D:\Develop\Remote-Sensing-Image-Wheat-Field-Segmentation\Remote-Sensing-Image-Wheat-Field-Segmentation\data_set\labelme_output', help='测试图像目录')
    parser.add_argument('--output', type=str, default=r'D:\Develop\Remote-Sensing-Image-Wheat-Field-Segmentation\Remote-Sensing-Image-Wheat-Field-Segmentation\predictions', help='输出目录')
    parser.add_argument('--threshold', type=float, default=0.5, help='阈值 (默认 0.5)')

    args = parser.parse_args()

    test_model(
        model_path=args.model,
        images_dir=args.images,
        output_dir=args.output,
        threshold=args.threshold
    )
