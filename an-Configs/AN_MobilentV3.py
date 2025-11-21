# ============================================================
# AN_MobilentV3.py -- Lightweight mobile segmentation model
# Based on MobileNetV3-Small + LRASPP for mobile deployment
# ============================================================

# ---- dataset suffix (EDIT HERE if needed)
IMG_SUFFIX = '.png'   # 改成你的真实图片后缀：'.png' / '.jpeg' ...
SEG_SUFFIX = '.png'   # 改成你的真实标注后缀：'.png' / '.bmp' ...

_FINAL_SIZE = (512, 512)
_NUM_CLASSES = 2
_DATA_ROOT = 'datasets/tongue_seg_v1/'
_DATASET_TYPE = 'ZihaoDataset'
_MAX_EPOCH = 300

_DATA_PREPROCESSOR = dict(
    type='SegDataPreProcessor',
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    pad_val=0,
    seg_pad_val=255,
    size_divisor=32,   # ← 关键：只设这个，别设 size
)

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
)

load_from = None
resume = False
log_level = 'INFO'
log_processor = dict(by_epoch=True)

# ---- Hooks & Visualizer
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1, max_keep_ckpts=30, save_best='mDice'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook', draw=False, interval=0),
)
visualizer = dict(
    type='SegLocalVisualizer',
    name='visualizer',
    vis_backends=[dict(type='LocalVisBackend')],
)

# ---- Model: MobileNetV3-Large + LRASPP (平衡精度与模型大小)
# 使用预训练权重 + 更大的模型容量以提高精度
_NORM_CFG = dict(type='BN', eps=0.001, requires_grad=True)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=_DATA_PREPROCESSOR,

    # MobileNetV3-Large backbone with pretrained weights
    backbone=dict(
        type='MobileNetV3',
        arch='large',  # 'large' ~5.5M params，精度更高
        out_indices=(1, 3, 16),  # large的输出层: [1]=stage1, [3]=stage2, [16]=stage5
        norm_cfg=_NORM_CFG,
    ),
    # 使用ImageNet预训练权重
    pretrained='open-mmlab://contrib/mobilenet_v3_large',

    # LRASPP decode head (增加channels提高容量)
    decode_head=dict(
        type='LRASPPHead',
        in_channels=(16, 24, 960),  # MobileNetV3-Large 的通道数 [stage1, stage2, stage5]
        in_index=(0, 1, 2),
        channels=256,  # 从128增加到256，提高模型容量
        input_transform='multiple_select',
        dropout_ratio=0.1,
        num_classes=_NUM_CLASSES,
        norm_cfg=_NORM_CFG,
        act_cfg=dict(type='ReLU'),
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, loss_name='loss_ce', avg_non_ignore=True),
            dict(type='DiceLoss', eps=1e-3, naive_dice=False, use_sigmoid=False, loss_weight=1.2, loss_name='loss_dice'),
            dict(type='LovaszLoss', loss_weight=0.5, loss_name='loss_lovasz', reduction='none'),  # 添加Lovasz loss提高边界精度
        ],
    ),

    # 辅助头提升精度
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=24,  # MobileNetV3-Large stage2 输出通道
        in_index=1,
        channels=64,  # 从32增加到64
        num_convs=2,  # 从1增加到2，增强特征提取
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=_NUM_CLASSES,
        norm_cfg=_NORM_CFG,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4, loss_name='loss_ce_aux', avg_non_ignore=True),
            dict(type='DiceLoss', eps=1e-3, naive_dice=False, use_sigmoid=False, loss_weight=0.4, loss_name='loss_dice_aux'),
        ],
    ),

    train_cfg=dict(),
    test_cfg=dict(mode='whole', crop_size=_FINAL_SIZE, stride=(256, 256)),
)

# ---- Optim & schedule (针对预训练模型优化)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-4, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        # 归一化与偏置不 decay
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        # backbone使用更小的学习率，因为是预训练的
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        }
    ),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=1/10, by_epoch=True, begin=0, end=20),  # 增加warmup epoch到20
    dict(type='PolyLR', eta_min=1e-6, power=0.9, begin=20, end=_MAX_EPOCH, by_epoch=True),
]

# ---- Loops
randomness = dict(seed=0)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=_MAX_EPOCH, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ---- Pipelines (保持与原配置一致的数据增强)
_LONG_EDGE = 1280

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', keep_ratio=True, scale=(_LONG_EDGE, _LONG_EDGE), interpolation='bilinear'),
    dict(type='RandomCrop', crop_size=_FINAL_SIZE, cat_max_ratio=0.94),
    dict(type='RandomRotate', prob=0.3, degree=5),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=0.1, contrast=0.1, saturation=0.1, hue=[0.001, 0.09], backend='pillow'),
    dict(
        type='GaussianBlur',
        magnitude_range=(0.3, 0.6),
        magnitude_std='inf',
        prob=0.5
    ),
    dict(type='PackSegInputs'),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', keep_ratio=True, scale=(_LONG_EDGE, _LONG_EDGE), interpolation='bilinear'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', keep_ratio=True, scale=(_LONG_EDGE, _LONG_EDGE), interpolation='bilinear'),
    dict(type='PackSegInputs'),
]

# ---- Dataloaders
train_dataloader = dict(
    batch_size=32, num_workers=8, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=_DATASET_TYPE,
        data_root=_DATA_ROOT,
        data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline,
        img_suffix=IMG_SUFFIX,
        seg_map_suffix=SEG_SUFFIX,
        reduce_zero_label=False,
    ),
)
val_dataloader = dict(
    batch_size=1, num_workers=8, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=_DATASET_TYPE,
        data_root=_DATA_ROOT,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=val_pipeline,
        img_suffix=IMG_SUFFIX,
        seg_map_suffix=SEG_SUFFIX,
        reduce_zero_label=False,
    ),
)
test_dataloader = dict(
    batch_size=1, num_workers=8, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=_DATASET_TYPE,
        data_root=_DATA_ROOT,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline,
        img_suffix=IMG_SUFFIX,
        seg_map_suffix=SEG_SUFFIX,
        reduce_zero_label=False,
    ),
)

# ---- Metrics / TTA / workdir
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='TestTimeAug', transforms=[
        [
         dict(type='Resize', keep_ratio=True, scale_factor=0.8),
         dict(type='Resize', keep_ratio=True, scale_factor=1.0),
         dict(type='Resize', keep_ratio=True, scale_factor=1.2),
        ],
        [dict(type='RandomFlip', direction='horizontal', prob=0.0),
         dict(type='RandomFlip', direction='horizontal', prob=1.0)],
        [dict(type='PackSegInputs')],
    ]),
]
work_dir = './work_dirs/AN_MobilentV3'
