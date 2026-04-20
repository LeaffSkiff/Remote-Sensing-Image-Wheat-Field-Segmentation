"""
将 RGB 彩色标签转换为单通道类别索引
类别定义：
- 0: 背景 (0, 0, 0)
- 1: soil (255, 215, 0) 金色
- 2: water (255, 0, 0) 红色
- 3: road (0, 255, 0) 绿色
- 4: field (0, 0, 255) 蓝色
"""
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# 颜色到类别的映射
COLOR_MAP = {
    (0, 0, 0): 0,       # 背景
    (255, 215, 0): 1,   # soil
    (255, 0, 0): 2,     # water
    (0, 255, 0): 3,     # road
    (0, 0, 255): 4,     # field
}

def rgb_to_label(rgb_array):
    """将 RGB 图像转换为单通道标签"""
    h, w, _ = rgb_array.shape
    label = np.zeros((h, w), dtype=np.uint8)

    for color, class_id in COLOR_MAP.items():
        mask = np.all(rgb_array == np.array(color), axis=-1)
        label[mask] = class_id

    return label

def convert_directory(input_dir, output_dir):
    """批量转换目录下的所有标签图片"""
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir)
                   if f.endswith(('.png', '.jpg', '.jpeg'))]

    for filename in tqdm(image_files, desc="转换标签"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 读取 RGB 标签
        rgb_img = Image.open(input_path).convert('RGB')
        rgb_array = np.array(rgb_img)

        # 转换为单通道标签
        label_array = rgb_to_label(rgb_array)

        # 保存为 PNG（使用最近邻插值避免压缩伪影）
        label_img = Image.fromarray(label_array, mode='L')
        label_img.save(output_path)

    print(f"\n转换完成！共处理 {len(image_files)} 张图片")
    print(f"输出目录：{output_dir}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='将 RGB 彩色标签转换为单通道类别索引')
    parser.add_argument('--input', type=str,
                        default=r'data_set/pytorch_unet_dataset/masks',
                        help='输入目录（RGB 标签）')
    parser.add_argument('--output', type=str,
                        default=r'data_set/pytorch_unet_dataset/masks_single',
                        help='输出目录（单通道标签）')

    args = parser.parse_args()
    convert_directory(args.input, args.output)
