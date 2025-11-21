# ============================================================
# AN_MobilentV3_optim.py -- Optimized mobile segmentation model
# Key improvements: higher resolution (1024), optimized augmentation,
# better loss weighting, increased model capacity
# Target: Improve tongue foreground IoU, Acc, Dice while keeping model < 40MB
# ============================================================

# ---- dataset suffix (EDIT HERE if needed)
IMG_SUFFIX = '.png'   # 改成你的真实图片后缀：'.png' / '.jpeg' ...
SEG_SUFFIX = '.png'   # 改成你的真实标注后缀：'.png' / '.bmp' ...

# ===== 关键优化：提升分辨率从512到1024 =====
_FINAL_SIZE = (1024, 1024)  # 从512提升到1024，捕捉更多细节
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

# ---- Model: MobileNetV3-Large + LRASPP (增强版)
# ===== 优化：增加模型容量以提升精度，同时保持 < 40MB =====
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

    # ===== 优化：增加decode head容量 256→384 =====
    decode_head=dict(
        type='LRASPPHead',
        in_channels=(16, 24, 960),  # MobileNetV3-Large 的通道数 [stage1, stage2, stage5]
        in_index=(0, 1, 2),
        channels=384,  # 从256增加到384，进一步提高模型容量
        input_transform='multiple_select',
        dropout_ratio=0.1,
        num_classes=_NUM_CLASSES,
        norm_cfg=_NORM_CFG,
        act_cfg=dict(type='ReLU'),
        align_corners=False,
        # ===== 优化：调整损失权重，参考最佳配置 =====
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.8, loss_name='loss_ce', avg_non_ignore=True),  # 降低CE权重1.0→0.8
            dict(type='DiceLoss', eps=1e-3, naive_dice=False, use_sigmoid=False, loss_weight=1.2, loss_name='loss_dice'),  # 保持Dice权重1.2
            dict(type='LovaszLoss', loss_weight=0.3, loss_name='loss_lovasz', reduction='none'),  # 降低Lovasz权重0.5→0.3
        ],
    ),

    # ===== 优化：增加辅助头容量 64→96 =====
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=24,  # MobileNetV3-Large stage2 输出通道
        in_index=1,
        channels=96,  # 从64增加到96
        num_convs=2,  # 保持2层卷积
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=_NUM_CLASSES,
        norm_cfg=_NORM_CFG,
        align_corners=False,
        # ===== 优化：调整辅助头损失权重 =====
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.3, loss_name='loss_ce_aux', avg_non_ignore=True),  # 降低0.4→0.3
            dict(type='DiceLoss', eps=1e-3, naive_dice=False, use_sigmoid=False, loss_weight=0.4, loss_name='loss_dice_aux'),  # 保持0.4
        ],
    ),

    train_cfg=dict(),
    # ===== 优化：更新test_cfg以匹配1024分辨率 =====
    test_cfg=dict(mode='whole', crop_size=_FINAL_SIZE, stride=(768, 768)),  # stride从256→768
)

# ---- Optim & schedule
# ===== 优化：调整学习率策略 =====
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-4, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        # 归一化与偏置不 decay
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        # ===== 优化：提高backbone学习率 0.1→0.3，让其更好适应高分辨率 =====
        custom_keys={
            'backbone': dict(lr_mult=0.3, decay_mult=1.0),  # 从0.1提升到0.3
        }
    ),
)
# ===== 优化：缩短warmup从20→10 epochs，参考最佳配置 =====
param_scheduler = [
    dict(type='LinearLR', start_factor=1/10, by_epoch=True, begin=0, end=10),  # warmup从20降到10
    dict(type='PolyLR', eta_min=1e-6, power=0.9, begin=10, end=_MAX_EPOCH, by_epoch=True),
]

# ---- Loops
randomness = dict(seed=0)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=_MAX_EPOCH, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ---- Pipelines
# ===== 优化：提升训练分辨率 1280→1536，参考最佳配置 =====
_LONG_EDGE = 1536  # 从1280提升到1536

# ===== 优化：调整数据增强强度，参考最佳配置使用更轻的增强 =====
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', keep_ratio=True, scale=(_LONG_EDGE, _LONG_EDGE), interpolation='bilinear'),
    dict(type='RandomCrop', crop_size=_FINAL_SIZE, cat_max_ratio=0.94),
    dict(type='RandomRotate', prob=0.3, degree=5),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # ===== 优化：降低ColorJitter强度 0.1→0.09，参考最佳配置 =====
    dict(type='ColorJitter', brightness=0.09, contrast=0.09, saturation=0.09, hue=[0.001, 0.009], backend='pillow'),
    # ===== 优化：降低GaussianBlur强度 (0.3,0.6)→(0.2,0.5)，参考最佳配置 =====
    dict(
        type='GaussianBlur',
        magnitude_range=(0.2, 0.5),  # 从(0.3, 0.6)降到(0.2, 0.5)
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
# ===== 优化：因分辨率从512→1024，batch size需要从32→8 =====
train_dataloader = dict(
    batch_size=8, num_workers=8, persistent_workers=True,  # 从32降到8，适配更大分辨率
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
# ===== 优化：增强TTA策略，增加更多尺度 =====
tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='TestTimeAug', transforms=[
        [
         dict(type='Resize', keep_ratio=True, scale_factor=0.7),
         dict(type='Resize', keep_ratio=True, scale_factor=0.8),
         dict(type='Resize', keep_ratio=True, scale_factor=0.9),
         dict(type='Resize', keep_ratio=True, scale_factor=1.0),
         dict(type='Resize', keep_ratio=True, scale_factor=1.1),
         dict(type='Resize', keep_ratio=True, scale_factor=1.2),
         dict(type='Resize', keep_ratio=True, scale_factor=1.3),
        ],
        [dict(type='RandomFlip', direction='horizontal', prob=0.0),
         dict(type='RandomFlip', direction='horizontal', prob=1.0)],
        [dict(type='PackSegInputs')],
    ]),
]
work_dir = './work_dirs/AN_MobilentV3_optim'
