"""
使用训练好的模型进行预测
"""
import sys
from pathlib import Path

# 添加 MMSegmentation 到路径
MMSEG_ROOT = Path(r'D:\Develop\mmsegmentation')  # 修改为你的 mmsegmentation 安装路径
sys.path.insert(0, str(MMSEG_ROOT))

from mmseg.apis import inference_segmentor, init_segmentor
import cv2
import numpy as np

def predict_image(config_file, checkpoint_file, image_path, output_dir):
    """
    预测单张图片

    Args:
        config_file: 配置文件路径
        checkpoint_file: 模型权重文件路径
        image_path: 输入图像路径
        output_dir: 输出目录
    """
    # 初始化模型
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

    # 推理
    result = inference_segmentor(model, image_path)

    # 保存结果
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_name = Path(image_path).name
    output_path = output_dir / f'pred_{img_name}'

    # 可视化结果
    model.show_result(image_path, result, palette=[(0, 0, 0), (0, 255, 0)], out_file=str(output_path))
    print(f"预测结果已保存到：{output_path}")

def predict_folder(config_file, checkpoint_file, image_dir, output_dir):
    """
    预测整个文件夹的图片
    """
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))

    print(f"找到 {len(image_files)} 张图片")

    for img_path in image_files:
        predict_image(config_file, checkpoint_file, str(img_path), output_dir)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='小麦田分割预测')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--image', type=str, required=True, help='输入图像或文件夹')
    parser.add_argument('--output', type=str, default='predictions', help='输出目录')

    args = parser.parse_args()

    if Path(args.image).is_dir():
        predict_folder(args.config, args.checkpoint, args.image, args.output)
    else:
        predict_image(args.config, args.checkpoint, args.image, args.output)
