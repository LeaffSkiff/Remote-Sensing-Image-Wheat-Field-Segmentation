import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def simple_threshold_segmentation(image_path, threshold=100, save_path=None):
    # 读取RGB图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("图像读取失败，请检查路径")
    # 转为RGB格式（OpenCV默认BGR）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 简单用绿色通道做阈值分割（绿色植被较明显）
    green_channel = img_rgb[:, :, 1]

    # 应用阈值分割，生成二值图像
    _, binary_mask = cv2.threshold(green_channel, threshold, 255, cv2.THRESH_BINARY_INV)

    # 保存结果
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # OpenCV保存前需要转回BGR格式或者直接保存单通道
        cv2.imwrite(save_path, binary_mask)
        print(f"分割结果已保存到: {save_path}")

    # 显示结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("原图")
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"阈值分割结果（阈值={threshold}）")
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')

    plt.show()

    return binary_mask


if __name__ == "__main__":
    image_path = "../data_0/test/tile_181.png"
    save_path = "results/threshold_segmentation.png"
    mask = simple_threshold_segmentation(image_path, threshold=180, save_path=save_path)

