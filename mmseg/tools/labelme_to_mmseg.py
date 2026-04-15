"""
将 labelme 标注导出为 MMSegmentation 训练格式（mask 图像）
"""
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import yaml

# 类别定义
CLASS_TO_ID = {
    '_background_': 0,
    'field': 1,
}

def labelme_json_to_mask(json_path, output_mask_path, img_size=None):
    """
    将 labelme JSON 转换为 mask 图像

    Args:
        json_path: labelme JSON 文件路径
        output_mask_path: 输出 mask 路径
        img_size: (width, height)，如果为 None 则从 JSON 读取
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    width = data.get('imageWidth')
    height = data.get('imageHeight')

    if img_size:
        width, height = img_size

    # 创建空白 mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # 解析 shapes
    for shape in data.get('shapes', []):
        label = shape['label']
        points = shape['points']
        shape_type = shape.get('shape_type', 'polygon')

        if label not in CLASS_TO_ID:
            print(f"警告：未知类别 {label}")
            continue

        class_id = CLASS_TO_ID[label]

        if shape_type == 'polygon':
            # 多边形
            pts = np.array(points, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], class_id)

        elif shape_type == 'rectangle':
            # 矩形
            pts = np.array(points, dtype=np.int32)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            mask[y_min:y_max, x_min:x_max] = class_id

    # 保存 mask
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)

    # 保存为 PNG（0=黑色，1=灰色）
    mask_img = Image.fromarray(mask * 255)
    mask_img.save(output_mask_path)

    print(f"已生成 mask: {output_mask_path}")
    return mask


def labelme_folder_to_mmseg(labelme_dir, output_img_dir, output_mask_dir):
    """
    批量处理 labelme 文件夹

    Args:
        labelme_dir: labelme JSON 目录（每个 JSON 在子文件夹中）
        output_img_dir: 输出图像目录
        output_mask_dir: 输出 mask 目录
    """
    labelme_dir = Path(labelme_dir)
    output_img_dir = Path(output_img_dir)
    output_mask_dir = Path(output_mask_dir)

    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有 JSON
    json_files = list(labelme_dir.rglob('*.json'))

    print(f"找到 {len(json_files)} 个 labelme JSON 文件")

    for json_path in json_files:
        # 跳过 info 文件
        if 'info.yaml' in str(json_path) or json_path.name.startswith('.'):
            continue

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 获取图像名
            img_name = data.get('imagePath')
            if not img_name:
                img_name = json_path.stem + '.png'

            # 找到原始图像（在同一目录或父目录）
            img_path = json_path.parent / img_name
            if not img_path.exists():
                # 尝试在上级目录找
                img_path = json_path.parent.parent / img_name

            if not img_path.exists():
                print(f"警告：找不到图像 {img_name}")
                continue

            # 复制图像
            import shutil
            shutil.copy(str(img_path), str(output_img_dir / img_name))

            # 生成 mask
            mask_path = output_mask_dir / img_name
            labelme_json_to_mask(str(json_path), str(mask_path))

        except Exception as e:
            print(f"处理 {json_path} 失败：{e}")

    print(f"\n完成!")
    print(f"图像输出：{output_img_dir}")
    print(f"Mask 输出：{output_mask_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='labelme 转 MMSegmentation 格式')
    parser.add_argument('--input', type=str, required=True, help='labelme JSON 目录')
    parser.add_argument('--img-out', type=str, required=True, help='输出图像目录')
    parser.add_argument('--mask-out', type=str, required=True, help='输出 mask 目录')

    args = parser.parse_args()

    labelme_folder_to_mmseg(args.input, args.img_out, args.mask_out)
