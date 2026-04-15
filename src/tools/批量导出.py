import os
import json
import numpy as np
from PIL import Image
import cv2
import shutil

# labelme 5.x 兼容的 JSON 导出脚本

json_dir = r"D:\Develop\Remote-Sensing-Image-Wheat-Field-Segmentation\Remote-Sensing-Image-Wheat-Field-Segmentation\data_set\labelme_annotations"
output_dir = r"D:\Develop\Remote-Sensing-Image-Wheat-Field-Segmentation\Remote-Sensing-Image-Wheat-Field-Segmentation\data_set\labelme_output"

os.makedirs(output_dir, exist_ok=True)

# labelme 环境 Python
python_exe = r"C:\Users\28027\miniconda3\envs\labelme\python.exe"

# 类别定义
CLASS_TO_ID = {
    '_background_': 0,
    'field': 1,
}

def json_to_mask(json_path, output_dir):
    """
    将 labelme JSON 转换为 mask 图像
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    width = data.get('imageWidth')
    height = data.get('imageHeight')

    if not width or not height:
        print(f"跳过 {json_path}: 缺少图像尺寸信息")
        return

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
    mask_img = Image.fromarray(mask * 255)
    mask_img.save(os.path.join(output_dir, 'label.png'))


for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        json_path = os.path.join(json_dir, filename)
        out_folder = os.path.join(output_dir, filename.replace(".json", ""))
        os.makedirs(out_folder, exist_ok=True)

        print(f"处理：{filename}")
        json_to_mask(json_path, out_folder)

        # 复制原图到输出目录
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            img_name = data.get('imagePath')
            if img_name:
                src_img = os.path.join(os.path.dirname(json_path), img_name)
                if os.path.exists(src_img):
                    shutil.copy(src_img, os.path.join(out_folder, 'img.png'))

print("完成!")
