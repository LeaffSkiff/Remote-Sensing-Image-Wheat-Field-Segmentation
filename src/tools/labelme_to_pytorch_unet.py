"""
将 labelme 格式的数据转换为 PyTorch-UNet 的 DataLoader 目录结构

源结构:
- labelme_annotations/: tile_000.png, tile_001.png, ... (切割好的图像)
- labelme_output/: tile_000/label.png, tile_001/label.png, ... (标注 mask)

目标结构:
- output/
  ├── imgs/    # 训练图像
  │   ├── tile_000.png
  │   ├── tile_001.png
  │   └── ...
  └── masks/   # 对应的标注 mask
      ├── tile_000.png
      ├── tile_001.png
      └── ...
"""

import os
import shutil
from pathlib import Path


def convert_to_pytorch_unet_format(
    images_source: str,
    masks_source: str,
    output_dir: str,
    split: str = "train"
):
    """
    转换 labelme 数据为 PyTorch-UNet 格式

    Args:
        images_source: labelme_annotations 目录路径（包含 tile_XXX.png 图像）
        masks_source: labelme_output 目录路径（包含 tile_XXX/label.png 标注）
        output_dir: 输出目录
        split: 数据集划分 (train/val/test)，用于创建子目录
    """
    images_source = Path(images_source)
    masks_source = Path(masks_source)
    output_dir = Path(output_dir)

    # 创建输出目录
    imgs_dir = output_dir / "imgs"
    masks_dir = output_dir / "masks"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有可用的图像文件
    image_files = list(images_source.glob("*.png"))
    image_files.extend(images_source.glob("*.jpg"))
    image_files.extend(images_source.glob("*.jpeg"))

    print(f"找到 {len(image_files)} 张图像")

    copied_count = 0
    for img_path in image_files:
        img_name = img_path.stem  # 例如 tile_000
        mask_folder = masks_source / img_name
        mask_path = mask_folder / "label.png"

        # 检查对应的 mask 是否存在
        if not mask_path.exists():
            print(f"警告：未找到 {img_name} 对应的 mask，跳过")
            continue

        # 复制图像
        dst_img = imgs_dir / img_path.name
        shutil.copy2(img_path, dst_img)

        # 复制 mask（重命名为与图像同名）
        dst_mask = masks_dir / f"{img_name}.png"
        shutil.copy2(mask_path, dst_mask)

        copied_count += 1

    print(f"\n转换完成!")
    print(f"已复制 {copied_count} 对图像 - mask 到：{output_dir}")
    print(f"  - 图像目录：{imgs_dir}")
    print(f"  - Mask 目录：{masks_dir}")


if __name__ == "__main__":
    # 默认路径配置
    BASE_DIR = Path(__file__).parent.parent.parent / "data_set"

    # 源目录
    images_source = BASE_DIR / "labelme_annotations"  # 切割好的图像
    masks_source = BASE_DIR / "labelme_output"        # labelme 导出的标注

    # 输出目录
    output_dir = BASE_DIR / "pytorch_unet_dataset"

    print("=" * 60)
    print("Labelme 转 PyTorch-UNet 数据集格式")
    print("=" * 60)
    print(f"\n源图像目录：{images_source}")
    print(f"源标注目录：{masks_source}")
    print(f"输出目录：{output_dir}")
    print()

    convert_to_pytorch_unet_format(
        images_source=images_source,
        masks_source=masks_source,
        output_dir=output_dir
    )

    print("\n" + "=" * 60)
    print("使用示例:")
    print("=" * 60)
    print("""
# 在训练脚本中使用:
from torch.utils.data import DataLoader
from src.ClassicUnet.Pytorch-UNet.utils.data_loading import BasicDataset

dataset = BasicDataset(
    images_dir='data_set/pytorch_unet_dataset/imgs',
    mask_dir='data_set/pytorch_unet_dataset/masks'
)
loader = DataLoader(dataset, batch_size=4, shuffle=True)
""")
