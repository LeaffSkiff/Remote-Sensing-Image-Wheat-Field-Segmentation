"""
准备 MMSegmentation 格式的数据集
将 train_dataset 中的数据复制到 data/wheat_field 目录下
"""
import os
import shutil
from pathlib import Path

# 源目录
SRC_ROOT = Path(__file__).parent.parent / 'train_dataset' / 'train' / 'data'
SRC_IMGS = SRC_ROOT / 'imgs'
SRC_MASKS = SRC_ROOT / 'masks'
SRC_TEST = SRC_ROOT / 'test'

# 目标目录 (MMSegmentation 格式)
DST_ROOT = Path(__file__).parent.parent.parent / 'data' / 'wheat_field'
DST_IMG_TRAIN = DST_ROOT / 'images' / 'train'
DST_IMG_VAL = DST_ROOT / 'images' / 'val'
DST_ANN_TRAIN = DST_ROOT / 'annotations' / 'train'
DST_ANN_VAL = DST_ROOT / 'annotations' / 'val'

def prepare_data():
    # 创建目录
    for d in [DST_IMG_TRAIN, DST_IMG_VAL, DST_ANN_TRAIN, DST_ANN_VAL]:
        d.mkdir(parents=True, exist_ok=True)

    # 获取所有图像文件
    img_files = sorted([f for f in SRC_IMGS.glob('*.png')])
    mask_files = sorted([f for f in SRC_MASKS.glob('*.png')])

    print(f"找到 {len(img_files)} 张训练图像")
    print(f"找到 {len(mask_files)} 张标注 mask")

    # 按 8:2 划分训练集和验证集
    split_idx = int(len(img_files) * 0.8)
    train_imgs = img_files[:split_idx]
    val_imgs = img_files[split_idx:]

    print(f"训练集：{len(train_imgs)} 张")
    print(f"验证集：{len(val_imgs)} 张")

    # 复制训练集
    for img_path in train_imgs:
        name = img_path.name
        # 复制图像
        shutil.copy(img_path, DST_IMG_TRAIN / name)
        # 复制对应 mask
        shutil.copy(SRC_MASKS / name, DST_ANN_TRAIN / name)

    # 复制验证集
    for img_path in val_imgs:
        name = img_path.name
        shutil.copy(img_path, DST_IMG_VAL / name)
        shutil.copy(SRC_MASKS / name, DST_ANN_VAL / name)

    # 处理测试集 (可选，用于最终预测)
    if SRC_TEST.exists():
        DST_IMG_TEST = DST_ROOT / 'images' / 'test'
        DST_IMG_TEST.mkdir(parents=True, exist_ok=True)
        test_files = list(SRC_TEST.glob('*.png'))
        for img_path in test_files:
            shutil.copy(img_path, DST_IMG_TEST / img_path.name)
        print(f"测试集：{len(test_files)} 张")

    print(f"\n数据准备完成！")
    print(f"目标目录：{DST_ROOT}")

if __name__ == '__main__':
    prepare_data()
