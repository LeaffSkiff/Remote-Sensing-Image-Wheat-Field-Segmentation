"""
将 MMSegmentation 预测结果转换为 labelme 格式
可以在 labelme 中打开进行微调
"""
import os
import json
import base64
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

# 类别定义
CLASSES = ['_background_', 'field']
CLASS_COLORS = {
    0: (0, 0, 0),      # 背景 - 黑色
    1: (0, 255, 0),    # 小麦田 - 绿色
}

def mask_to_labelme_json(mask, image_path, output_json_path):
    """
    将分割 mask 转换为 labelme JSON 格式

    Args:
        mask: 预测的 mask (H, W)，值 0=背景，1=小麦田
        image_path: 原始图像路径
        output_json_path: 输出的 JSON 路径
    """
    # 读取原始图像
    img = np.array(Image.open(image_path).convert('RGB'))
    height, width = img.shape[:2]

    # 将 mask 转为 contours（多边形）
    # 这样可以 labelme 中编辑
    polygons = mask_to_polygons(mask)

    # 构建 labelme JSON
    json_data = {
        'version': '5.8.1',
        'flags': {},
        'shapes': [],
        'imagePath': os.path.basename(image_path),
        'imageData': encode_image_to_base64(image_path),
        'imageHeight': height,
        'imageWidth': width,
    }

    # 添加多边形
    for poly in polygons:
        json_data['shapes'].append({
            'label': 'field',
            'points': poly,
            'group_id': None,
            'shape_type': 'polygon',
            'flags': {}
        })

    # 保存 JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"已保存：{output_json_path}")
    return json_data


def mask_to_polygons(mask, min_area=100, epsilon_factor=0.004):
    """
    将 mask 转换为多边形列表

    Args:
        mask: 二值 mask，目标区域为 1
        min_area: 最小多边形面积
        epsilon_factor: 多边形简化系数

    Returns:
        list of polygons, each polygon is [[x1,y1], [x2,y2], ...]
    """
    polygons = []

    # 转 uint8
    mask_uint8 = (mask * 255).astype(np.uint8)

    # 查找轮廓
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 过滤小区域
        if cv2.contourArea(contour) < min_area:
            continue

        # 简化多边形
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 转为 labelme 格式 [[x1,y1], [x2,y2], ...]
        polygon = []
        for point in approx:
            x, y = point[0]
            polygon.append([float(x), float(y)])

        if len(polygon) >= 3:  # 至少 3 个点
            polygons.append(polygon)

    return polygons


def encode_image_to_base64(image_path):
    """将图像编码为 base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def process_folder(pred_mask_dir, original_image_dir, output_labelme_dir):
    """
    批量处理文件夹

    Args:
        pred_mask_dir: 预测 mask 目录
        original_image_dir: 原始图像目录
        output_labelme_dir: 输出 labelme JSON 目录
    """
    pred_mask_dir = Path(pred_mask_dir)
    original_image_dir = Path(original_image_dir)
    output_labelme_dir = Path(output_labelme_dir)

    output_labelme_dir.mkdir(parents=True, exist_ok=True)

    # 处理所有 mask
    mask_files = list(pred_mask_dir.glob('*.png')) + list(pred_mask_dir.glob('*.jpg'))

    print(f"找到 {len(mask_files)} 个预测文件")

    for mask_path in mask_files:
        # 读取 mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # 转为 0/1
        mask = (mask > 128).astype(np.uint8)

        # 找到对应的原始图像
        image_path = original_image_dir / mask_path.name.replace('.png', '.png')
        if not image_path.exists():
            # 尝试其他格式
            image_path = original_image_dir / mask_path.name.replace('.png', '.jpg')

        if not image_path.exists():
            print(f"警告：找不到原始图像 {mask_path.name}")
            continue

        # 生成 JSON
        output_json = output_labelme_dir / mask_path.stem / f"{mask_path.stem}.json"

        try:
            mask_to_labelme_json(mask, str(image_path), str(output_json))

            # 同时复制原始图像到同一目录（方便 labelme 查看）
            img_output_dir = output_labelme_dir / mask_path.stem
            img_output_dir.mkdir(exist_ok=True)
            import shutil
            shutil.copy(image_path, img_output_dir / image_path.name)

        except Exception as e:
            print(f"处理 {mask_path} 失败：{e}")

    print(f"\n完成！输出目录：{output_labelme_dir}")
    print(f"可以在 labelme 中打开生成的 JSON 文件进行微调")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='预测结果转 labelme 格式')
    parser.add_argument('--pred', type=str, required=True, help='预测 mask 目录')
    parser.add_argument('--image', type=str, required=True, help='原始图像目录')
    parser.add_argument('--output', type=str, required=True, help='输出 labelme JSON 目录')

    args = parser.parse_args()

    process_folder(args.pred, args.image, args.output)
