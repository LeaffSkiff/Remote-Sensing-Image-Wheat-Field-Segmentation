import os
from tkinter import image_names

import matplotlib.pyplot as plt
from PIL import Image
from sympy import is_amicable
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import torch

#from my_unet_v1.train import images


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_transform = mask_transform
        #transform 可以对数据进行预处理

        # 获取所有 mask 文件的基本名称（不带扩展名）
        mask_names = []
        for f in os.listdir(masks_dir):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                mask_names.append(f)
        mask_names.sort()

        # 匹配图像和标签：通过基本名称匹配（支持不同扩展名）
        self.image_names = []
        for img_file in os.listdir(images_dir):
            if not img_file.endswith(('.png', '.jpg', '.jpeg')):
                continue
            # 获取基本名称（去掉扩展名）
            base_name = os.path.splitext(img_file)[0]
            # 在 mask 目录中查找同名的文件（支持.png/.jpg 等不同扩展名）
            for mask_file in os.listdir(masks_dir):
                if os.path.splitext(mask_file)[0] == base_name:
                    self.image_names.append(img_file)
                    break

        self.image_names.sort()

        if len(self.image_names) == 0:
            print(f"警告：在 {images_dir} 和 {masks_dir} 中没有找到匹配的图像 - 标签对")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_file = self.image_names[idx]
        base_name = os.path.splitext(img_file)[0]

        image_path = os.path.join(self.images_dir, img_file)
        # 查找对应的 mask 文件
        mask_path = None
        for f in os.listdir(self.masks_dir):
            if os.path.splitext(f)[0] == base_name:
                mask_path = os.path.join(self.masks_dir, f)
                break

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 转为单通道灰度图

        if self.transform:
            image = self.transform(image)
        # 直接使用 numpy 数组转换，避免 ToTensor 的归一化
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask

class TestDataset(Dataset):
    """
    测试数据集 - 支持两种目录结构:
    1. 扁平结构：images_dir/tile_001.png, images_dir/tile_002.png, ...
    2. 文件夹结构：images_dir/tile_001/img.png, images_dir/tile_002/img.png, ...
    """
    def __init__(self, images_dir, masks_dir=None, transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_transform = mask_transform

        # 检查是否是文件夹结构 (每个子文件夹包含 img.png 和 label.png)
        self.is_folder_structure = False
        self.image_folders = []

        for name in os.listdir(images_dir):
            folder_path = os.path.join(images_dir, name)
            if os.path.isdir(folder_path):
                img_path = os.path.join(folder_path, 'img.png')
                if os.path.exists(img_path):
                    self.is_folder_structure = True
                    self.image_folders.append(name)

        if self.is_folder_structure:
            self.image_folders.sort()
        else:
            # 扁平结构
            self.image_names = [f for f in os.listdir(images_dir)
                                if f.endswith(('.png', '.jpg', '.jpeg'))]
            self.image_names.sort()

    def __len__(self):
        if self.is_folder_structure:
            return len(self.image_folders)
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.is_folder_structure:
            # 文件夹结构：folder/img.png
            folder_name = self.image_folders[idx]
            folder_path = os.path.join(self.images_dir, folder_name)
            image_path = os.path.join(folder_path, 'img.png')

            # 如果有 masks_dir，读取对应的 label.png
            if self.masks_dir is not None:
                mask_path = os.path.join(self.masks_dir, folder_name, 'label.png')
            else:
                # 尝试在同一文件夹中找 label.png
                mask_path = os.path.join(folder_path, 'label.png')
                if not os.path.exists(mask_path):
                    mask_path = None
        else:
            # 扁平结构
            image_path = os.path.join(self.images_dir, self.image_names[idx])
            if self.masks_dir is not None:
                mask_path = os.path.join(self.masks_dir, self.image_names[idx])
            else:
                mask_path = None

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            if self.mask_transform:
                mask = self.mask_transform(mask)
            return image, mask

        return image, image_path


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),  # 把图像转换为一个张量，同时值变为 0-1
        transforms.Normalize([0.5], [0.5])  # 归一化
    ])

    mask_transform = transforms.ToTensor()
    # 转换成了一个张量
    tmp = SegmentationDataset(
        images_dir='../../train_dataset/train/imgs',
        masks_dir='../../train_dataset/train/masks',
        transform=transform,
        mask_transform=mask_transform
    )

    print(len(tmp))

    _, img = tmp[1]
    img = img.squeeze()
    plt.figure("Image")  # 图像窗口名称
    plt.imshow(img)
    plt.axis('off')  # 关掉坐标轴为 off
    plt.title('image')  # 图像题目
    plt.show()
