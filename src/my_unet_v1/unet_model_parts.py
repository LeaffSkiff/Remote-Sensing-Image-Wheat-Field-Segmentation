import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    #这里定义了一个双卷积结构，首先双卷积结构能够更好的进行特征提取，
    # 第二双卷积可以用更小的算力提升感受野，
    # 第三每次卷积后加 BatchNorm 和激活函数，可以让训练过程更稳定，加速收敛
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),#二维卷积
            nn.BatchNorm2d(mid_channels),  #归一化操作，加快收敛
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv.forward(x)

class Down(nn.Module):
    #下采样模块，最大池化实现
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x_conv = self.conv(x)      # 卷积后的特征，用于跳跃连接
        x_pool = self.pool(x_conv) # 池化后的特征，送给下一层下采样
        return x_conv, x_pool

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            #bilinear双线性插值无需训练速度较快
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            #看不懂为什么中间层//2。说是减少参数量
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            #这个里面用的就是转置卷积了。
            # 卷积操作实际上可以看作矩阵相乘。一个原始的矩阵乘以卷积矩阵，得到一个小矩阵。
            # 卷积核可以进行重新排列的到一个卷积矩阵，使得最终得到的结果是一样的。
            # 反过来既然卷积核重排列得到的卷积矩阵，矩阵相乘也能得到比原来大的结果
            # 当然，这样的卷积核一定是比原来的输入矩阵要大的
            #我理解的实际上就是一个比原来输入矩阵要大的卷积核同样的进行卷积操作，然后对输入做填充，填充到
            # 能做出输出的大矩阵，然后进行卷积。

    def forward(self, x1, x2):
        #x1是上采样输入，x2是跳跃链接特征图（左边copy-crop来的）
        x1 = self.up(x1)
        #进行一次上采样操作得到输出
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        #对齐尺寸，这里用的是0填充（论文中进行镜像填充或者crop裁剪）
        x = torch.cat([x2, x1], dim=1)
        #进行通道拼接，dim代表在第几纬度上进行拼接
        return self.conv(x)
        #上采样完了，进行拼接，然后再转置卷积完成forward操作

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    #看起来没卷积，实际上对通道数进行了变换
    def forward(self, x):
        return self.conv(x)

