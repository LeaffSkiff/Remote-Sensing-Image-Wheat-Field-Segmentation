import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from unet_model import Unet
from dataset import SegmentationDataset
import tqdm

transform = transforms.Compose([
    transforms.ToTensor(),  # 把图像转换为一个张量，同时值变为0-1
    transforms.Normalize([0.5], [0.5])  # 归一化
])

mask_transform = transforms.ToTensor()
#转换成了一个张量
train_dataset = SegmentationDataset(
    images_dir='../../train_dataset/train/imgs',
    masks_dir='../../train_dataset/train/masks',
    transform=transform,
    mask_transform=mask_transform
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
#dataLoader可以将自定义的数据集对象进行打包。
# batch_size每次数量，shuffle每个epoch进行打乱避免出现记忆，num_workers子线程

# 实际上是一个张量，张量(batch, channels, height, width)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Unet(3, 1).to(device)
criterion = nn.BCEWithLogitsLoss()
#损失函数：二分类交叉熵损失+sigmoid激活
# 相当于sigmoid = torch.sigmoid(output)
# loss = F.binary_cross_entropy(sigmoid, target)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
#优化器，将model中的参数交给优化器处理，学习率为1e-3

epochs = 30

for epoch in tqdm.tqdm(range(epochs)):
    model.train()
    #切换为训练模式
    running_loss = 0.0
    #用于监控每个epoch总loss
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        #清空梯度避免累加
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")
    torch.save(model.state_dict(), f'./model/model_epoch_.pth')
#多线程训练需要在__name__ == '__main__'下操作