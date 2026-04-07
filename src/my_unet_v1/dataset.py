import os
from tkinter import image_names

import matplotlib.pyplot as plt
from PIL import Image
from sympy import is_amicable
from torch.utils.data import Dataset
import torchvision.transforms as transforms

#from my_unet_v1.train import images


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_transform = mask_transform
        #transform可以对数据进行预处理
        self.image_names = [f for f in os.listdir(images_dir)
                            if f.endswith(('.png', '.jpg', '.jpeg'))]
        #取数据存列表
        self.image_names.sort()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_names[idx])
        mask_path = os.path.join(self.masks_dir, self.image_names[idx])

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 转为单通道灰度图（黑白）

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = (mask > 0).float()

        return image, mask

class TestDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None, transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_names = [f for f in os.listdir(images_dir)
                            if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.image_names.sort()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_names[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.masks_dir is not None:
            mask_path = os.path.join(self.masks_dir, self.image_names[idx])
            mask = Image.open(mask_path).convert('L')  # 转为单通道灰度图（黑白）
            if self.mask_transform:
                mask = self.mask_transform(mask)
            return image, mask

        return image, image_path


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),  # 把图像转换为一个张量，同时值变为0-1
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