# 小麦田分割 - UNet 配置
_base_ = [
    '../_base_/models/unet_r50-d8.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

# 模型设置
crop_size = (512, 512)
model = dict(
    data_preprocessor=dict(
        size=crop_size,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255),
    decode_head=dict(
        num_classes=2,  # 背景 + 小麦田
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# 数据设置
data_root = 'data/wheat_field'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=180),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='annotations/train'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='images/val',
            seg_map_path='annotations/val'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# 评估设置
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator

# 训练设置
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005))

param_scheduler = [
    dict(type='PolyLR', eta_min=1e-4, power=0.9)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)

# 保存设置
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=4000, max_keep_ckpts=3, save_best='mIoU'))

random_seed = 42
