from dataset import TestDataset
from unet_model import Unet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm   #进度条模块
from PIL import Image
import numpy as np
import os
import cv2


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_dataset = TestDataset(
    images_dir='../../train_dataset/train/test',
    transform=transform
)

test_dataloader = DataLoader(test_dataset, batch_size=1)
model_path = './model/model_epoch_.pth'
save_predict = True
predict_dir = '../../model_predictions/test_result_1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Unet(3, 1)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model = model.eval()

with torch.no_grad():
    for image, path in tqdm(test_dataloader):
        image = image.to(device)
        output = model(image)
        output = torch.sigmoid(output)  # 对于二分类，用sigmoid
        output = (output > 0.85).float()
        if save_predict:
            # logits = output.detach()
            # print(
            #     f"Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
            # 将预测结果转为图像保存
            output_np = output.squeeze().cpu().numpy() * 255
            #对于张量，删除batch，channels等值为1的纬度
            output_img = Image.fromarray(output_np.astype(np.uint8))
            base_name = os.path.basename(path[0])
            name = os.path.splitext(base_name)[0] + '_pred.png'
            output_img.save(os.path.join(predict_dir, name))


print("测试完成，预测结果保存在", predict_dir, '目录下')