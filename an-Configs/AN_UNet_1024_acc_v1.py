# ============================================================
# AN_UNet_1024_acc_v1.py  -- Enhanced Data Augmentation for Tongue Segmentation
# ============================================================
#
# 改进点：
# 1. 使用 PhotoMetricDistortion 替换 ColorJitter（更适合医学图像）
# 2. 添加 CLAHE（对比度受限自适应直方图均衡）处理局部光照不均
# 3. 增大 RandomRotate 角度范围（5° → 15°）
# 4. 添加 AdjustGamma（伽马校正）模拟不同曝光
# 5. 添加 RandomCutOut（随机擦除）提高遮挡鲁棒性
# 6. 调整 GaussianBlur 概率，避免过度模糊
# 7. 添加 Pad + RandomCrop 组合增加位置多样性

# ---- dataset suffix (EDIT HERE if needed)
IMG_SUFFIX = '.png'   # 改成你的真实图片后缀：'.png' / '.jpeg' ...
SEG_SUFFIX = '.png'   # 改成你的真实标注后缀：'.png' / '.bmp' ...

_FINAL_SIZE = (896, 896)
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

# ---- Hooks & Visualizer（mmseg 可视化 + mmcv 已安装）
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

# ---- Model
_NORM_CFG = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=_DATA_PREPROCESSOR,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        downsamples=(True, True, True, True),
        with_cp=False,
        norm_cfg=_NORM_CFG,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        conv_cfg=None,
        norm_eval=False,
    ),
    pretrained=None,
    decode_head=dict(
        type='FCNHead',
        in_channels=64, in_index=4, channels=64, num_convs=1,
        concat_input=False, dropout_ratio=0.1, num_classes=_NUM_CLASSES,
        norm_cfg=_NORM_CFG, align_corners=False, ignore_index=255,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.8, loss_name='loss_ce', avg_non_ignore=True),
            dict(type='DiceLoss', eps=1e-3, naive_dice=False, use_sigmoid=False, loss_weight=1.2, loss_name='loss_dice'),
        ],
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128, in_index=3, channels=64, num_convs=1,
        concat_input=False, dropout_ratio=0.1, num_classes=_NUM_CLASSES,
        norm_cfg=_NORM_CFG, align_corners=False, ignore_index=255,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.15, loss_name='loss_ce_aux', avg_non_ignore=True),
            dict(type='DiceLoss', eps=1e-3, naive_dice=False, use_sigmoid=False, loss_weight=0.25, loss_name='loss_dice_aux'),
        ],
    ),
    train_cfg=dict(),
	# TODO: 这里 mode:'slide' 还是 'whole'? 
    test_cfg=dict(mode='whole', crop_size=_FINAL_SIZE, stride=(672, 672)),
)

# ---- Optim & schedule
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-4, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(  # 归一化与偏置不 decay，常带来 0.1~0.3 mDice
        norm_decay_mult=0.0, bias_decay_mult=0.0),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=1/10, by_epoch=True, begin=0, end=10),
    dict(type='PolyLR', eta_min=1e-6, power=0.9, begin=10, end=_MAX_EPOCH, by_epoch=True),
]

# ---- Loops
randomness = dict(seed=0)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=_MAX_EPOCH, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ============================================================
# ---- Enhanced Pipelines（增强版数据增强）
# ============================================================
_LONG_EDGE = 1280  # 调整为 `_FINAL_SIZE=(896, 896)` 对应的长边

train_pipeline = [
    # ---- 1. 数据加载
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),

    # ---- 2. 尺度增强（模拟不同拍摄距离和舌头大小）
    dict(
        type='Resize',
        scale=(_LONG_EDGE, _LONG_EDGE),
        keep_ratio=True,
        interpolation='bilinear'
    ),
    # 注意: BioMedical3DPad 仅适用于3D医学图像，2D图像无需 Pad
    # Resize 后直接 RandomCrop 即可提供位置多样性
    dict(
        type='RandomCrop',
        crop_size=_FINAL_SIZE,
        cat_max_ratio=0.92  # 防止某个类别占比过大
    ),

    # ---- 3. 几何变换增强
    dict(
        type='RandomRotate',
        prob=0.5,           # 提高概率 0.3 → 0.5
        degree=10,          # 增大角度范围 5° → 10°，模拟更多拍摄角度
        pad_val=0,
        seg_pad_val=255
    ),
    dict(
        type='RandomFlip',
        prob=0.5,
        direction='horizontal'  # 舌头有左右对称性
    ),

    # ---- 4. 光度增强（重点优化）
    # 使用 PhotoMetricDistortion 替换 ColorJitter（更适合医学图像）
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,           # 亮度随机偏移 ±32
        contrast_range=(0.8, 1.2),     # 对比度 0.8~1.2 倍
        saturation_range=(0.8, 1.2),   # 饱和度 0.8~1.2 倍
        hue_delta=8                   # 色调偏移 ±8
    ),

    # 添加 CLAHE（对比度受限自适应直方图均衡）
    # 处理局部光照不均，医学图像常用
    dict(
        type='CLAHE',
        clip_limit=2.0,         # 对比度限制，防止过度增强
        tile_grid_size=(8, 8)   # 网格大小
    ),

    # 注意: AdjustGamma 在 MMSeg 中不支持随机 gamma，固定值无效果，故移除
    # 如需伽马增强，可使用 PhotoMetricDistortion 的亮度和对比度调整替代

    # ---- 5. 模糊增强（适度降低概率，避免丢失舌苔纹理）
    dict(
        type='GaussianBlur',
        magnitude_range=(0.1, 0.4),  # 降低模糊强度 0.2~0.5 → 0.1~0.4
        magnitude_std='inf',
        prob=0.3  # 降低概率 0.8 → 0.3
    ),

    # ---- 6. 遮挡增强（新增）
    # RandomCutOut：随机擦除图像区域，模拟舌头局部被遮挡
    dict(
        type='RandomCutOut',
        prob=0.15,                # 25% 概率应用 cutout
        n_holes=(1, 2),          # 擦除 1~3 个区域
        cutout_shape=(10, 10),   # 每个擦除区域大小 10×10
        fill_in=(0, 0, 0),       # 用黑色填充图像
        seg_fill_in=255,         # 用 255 (ignore_index) 填充标签
    ),

    dict(type='PackSegInputs'),
]

# 验证和测试 pipeline 保持简洁
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', keep_ratio=True, scale=(_LONG_EDGE, _LONG_EDGE), interpolation='bilinear'),
    # 注意：不需要手动 Pad！
    # SegDataPreProcessor 已配置 size_divisor=32，会自动 pad 到 32 的倍数
    # 同时处理图像 (pad_val=0) 和标签 (seg_pad_val=255)
    dict(type='PackSegInputs', meta_keys=('img_path', 'img_shape', 'pad_shape', 'scale_factor'))  # 不包含 'ori_shape'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', keep_ratio=True, scale=(_LONG_EDGE, _LONG_EDGE), interpolation='bilinear'),
    # 注意：不需要手动 Pad！
    # SegDataPreProcessor 已配置 size_divisor=32，会自动 pad 到 32 的倍数
    dict(type='PackSegInputs', meta_keys=('img_path', 'img_shape', 'pad_shape', 'scale_factor'))  # 不包含 'ori_shape'),
]

# ---- Dataloaders（显式写后缀，避免 data_list 为空）
train_dataloader = dict(
    batch_size=8, num_workers=8, persistent_workers=True,
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
    batch_size=1, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=_DATASET_TYPE,
        data_root=_DATA_ROOT,
        data_prefix=dict(img_path='img_dir/val_', seg_map_path='ann_dir/val_'),
        pipeline=val_pipeline,
        img_suffix=IMG_SUFFIX,
        seg_map_suffix=SEG_SUFFIX,
        reduce_zero_label=False,
    ),
)
test_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=_DATASET_TYPE,
        data_root=_DATA_ROOT,
        data_prefix=dict(img_path='img_dir/val_', seg_map_path='ann_dir/val_'),
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
        [dict(type='Resize', keep_ratio=True, scale_factor=0.75),
         dict(type='Resize', keep_ratio=True, scale_factor=1.0),
         dict(type='Resize', keep_ratio=True, scale_factor=1.25)],
        [dict(type='RandomFlip', direction='horizontal', prob=0.0),
         dict(type='RandomFlip', direction='horizontal', prob=1.0)],
        [dict(type='PackSegInputs')],
    ]),
]
work_dir = './work_dirs/AN_UNet_1024_acc_v1'
