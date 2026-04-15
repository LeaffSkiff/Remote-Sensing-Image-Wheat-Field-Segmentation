# MMSegmentation 小麦田分割实验指南

## 1. 环境验证

首先验证 MMSegmentation 是否正确安装：

```bash
# 进入你的 mmsegmentation 环境
conda activate mmsegmentation  # 或者你的环境名

# 验证安装
python -c "import mmseg; print(mmseg.__version__)"
```

## 2. 数据准备

你的数据已经在正确位置，需要组织成如下结构：

```
data/
└── wheat_field/
    ├── images/
    │   ├── train/
    │   └── val/
    └── annotations/
        ├── train/
        └── val/
```

运行以下命令准备数据（会自动复制/链接文件）：

```bash
python src/tools/prepare_mmseg_data.py
```

## 3. 训练模型

### 方式一：使用预定义模型（推荐新手）

```bash
# DeepLabV3+ (效果好，速度快)
python tools/train.py configs/deeplabv3/deeplabv3_r50-d8_512x512_40k_wheat.py

# UNet (经典分割网络)
python tools/train.py configs/unet/unet-r50-d8_512x512_40k_wheat.py

# PSPNet
python tools/train.py configs/pspnet/pspnet_r50-d8_512x512_40k_wheat.py
```

### 方式二：自定义配置训练

```bash
# 使用自定义配置文件
python tools/train.py configs/wheat/wheat_unet_custom.py --work-dir work_dirs/wheat_unet
```

### 多 GPU 训练（如果有多个显卡）

```bash
./tools/dist_train.sh configs/wheat/wheat_unet_custom.py 4 --work-dir work_dirs/wheat_unet
```

## 4. 评估模型

```bash
# 在验证集上评估
python tools/test.py configs/wheat/wheat_unet_custom.py work_dirs/wheat_unet/iter_40000.pth --eval mIoU

# 生成可视化结果
python tools/test.py configs/wheat/wheat_unet_custom.py work_dirs/wheat_unet/iter_40000.pth --show
```

## 5. 预测新图像

```bash
# 单张图片预测
python tools/inference.py configs/wheat/wheat_unet_custom.py work_dirs/wheat_unet/best.pth to_predict/image.png --output-dir predictions/
```

## 6. 配置文件说明

关键参数解释：

```python
# 模型 backbone
backbone=dict(type='ResNet', depth=50)  # 可选 ResNet-50/101, MobileNet, 等

# 类别数（背景 + 小麦田 = 2）
num_classes=2

# 输入尺寸
img_scale=(512, 512)

# 学习率
lr=0.01

# 批次大小
samples_per_gpu=4  # 每个 GPU 的 batch size

# 训练轮数
max_iters=40000  # 迭代次数
```

## 7. 实验建议

### 对比不同模型

| 模型 | 配置 | 特点 |
|------|------|------|
| UNet | `unet-r50-d8` | 经典，适合小数据集 |
| DeepLabV3+ | `deeplabv3_r50-d8` | 效果好，推荐 |
| PSPNet | `pspnet_r50-d8` | 多尺度信息好 |
| SegFormer | `segformer_mit-b0` | Transformer，效果最好 |

### 超参数调优

1. **学习率**: 尝试 0.001, 0.01, 0.1
2. **Batch size**: 根据显存调整 2/4/8
3. **数据增强**: 翻转、旋转、色彩抖动

## 8. 常见问题

### Q: 显存不足怎么办？
A: 减小 `samples_per_gpu` 或 `img_scale`

### Q: 训练太慢？
A: 使用更小的 backbone (MobileNet) 或减少 `max_iters`

### Q: mIoU 太低？
A: 
- 检查数据标注质量
- 增加训练轮数
- 尝试更强的 backbone
- 调整学习率

## 9. 输出位置

训练输出在 `work_dirs/` 目录：
- `best.pth` - 最佳模型权重
- `latest.pth` - 最后一次迭代的权重
- `log.txt` - 训练日志
- `visualizations/` - 可视化结果
