import torch
import torch.nn as nn
import torch.nn.functional as f
from unet_model_parts import *

class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=5, bilinear=True):
        #out输出，可以是n类
        super().__init__()
        self.bilinear = bilinear

        self.down1 = Down(in_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        factor = 2 if bilinear else 1
        self.bottom = DoubleConv(512, 1024//factor)
        # 底部瓶颈层

        self.up1 = Up(1024, 512//factor, self.bilinear)
        self.up2 = Up(512, 256//factor, self.bilinear)
        self.up3 = Up(256, 128//factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        #当进行线性插值时，需要手动调整通道，使得后续进行torch.cat拼接操作后能对得上
        #当进行转置卷积时，转置卷积计算后通道自动调整方便cat
        self.out = OutConv(64, out_channels)
        #outchannels其实是class

    def forward(self, x):
        x1, x1_pool = self.down1(x)
        x2, x2_pool = self.down2(x1_pool)
        x3, x3_pool = self.down3(x2_pool)
        x4, x4_pool = self.down4(x3_pool)

        x5 = self.bottom(x4_pool)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out(x)
        return logits

    #def use_checkpointing(self):
    # 内存优化函数