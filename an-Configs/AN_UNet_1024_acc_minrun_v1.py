# ============================================================
# AN_UNet_1024_acc_minrun.py  -- minimal, no-Albu, explicit suffixes
# ============================================================

# ---- dataset suffix (EDIT HERE if needed)
IMG_SUFFIX = '.png'   # 改成你的真实图片后缀：'.png' / '.jpeg' ...
SEG_SUFFIX = '.png'   # 改成你的真实标注后缀：'.png' / '.bmp' ...

_FINAL_SIZE = (1024, 1024)
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
    test_cfg=dict(mode='whole', crop_size=_FINAL_SIZE, stride=(768, 768)),
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

# ---- Pipelines（纯 mmseg 增强，去掉 Albu）
_LONG_EDGE = 1536  # 显存不够可降到 1280

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', keep_ratio=True, scale=(_LONG_EDGE, _LONG_EDGE), interpolation='bilinear'),
    dict(type='RandomCrop', crop_size=_FINAL_SIZE, cat_max_ratio=0.75),
	dict(type='RandomRotate', prob=0.3, degree=5),   # 面部轻微转头常见
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=0.09, contrast=0.09, saturation=0.09, hue=[0.001, 0.009], backend='pillow'),
    # dict(type='GaussianBlur', ksize=3, sigma_min=0.1, sigma_max=0.3, prob=0.2),
	dict(
        type='GaussianBlur',
        magnitude_range=(0.2, 0.5),  # 较轻的模糊程度
        magnitude_std='inf',
        prob=0.5  # 50%概率应用模糊
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

# ---- Dataloaders（显式写后缀，避免 data_list 为空）
train_dataloader = dict(
    batch_size=6, num_workers=8, persistent_workers=True,
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
        data_prefix=dict(img_path='img_dir/val_', seg_map_path='ann_dir/val_'),
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
work_dir = './work_dirs/AN_UNet_1024_acc_minrun_v1'
