# AN_UNet_1024.py
_FINAL_SIZE = (1024, 1024)
# XXX: 这里是分割的类别数
_NUM_CLASSES = 2
# XXX: 这里是数据集的根目录和类型
_DATA_ROOT = 'datasets/tongue_seg_v0/'
_DATASET_TYPE = 'ZihaoDataset'
_MAX_EPOCH = 350  # 训练更聚焦，后期靠EMA/挑最优；想更久再拉高

_DATA_PREPROCESSOR = dict(
    type='SegDataPreProcessor',
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    pad_val=0,          # pad 图像像素用 0（黑）
    seg_pad_val=255     # pad 掩膜用 255（= ignore_index），不要设 0
)

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))

load_from = None
resume = False
log_level = 'INFO'
log_processor = dict(by_epoch=True)

# ========= Hooks =========
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        max_keep_ckpts=30,
        save_best='mDice',   # 二分类更关注Dice
        type='CheckpointHook'),
    logger=dict(interval=1, log_metric_by_epoch=True, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    # visualization=dict(type='SegVisualizationHook')
	visualization=dict(type='VisualizationHook')  # mmengine 的，不会 import mmseg 可视化链
)

# ========= Model =========
_NORM_CFG = dict(requires_grad=True, type='BN')

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

    # Cityscapes 的 FCN UNet 权重与当前头部/类别不匹配，别强塞
    # init_cfg=dict(type='Pretrained', checkpoint='...')  # 如需自有权重再打开
    pretrained=None,

    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=_NUM_CLASSES,
        norm_cfg=_NORM_CFG,
        align_corners=False,
        ignore_index=255,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, loss_name='loss_ce'),
            dict(type='DiceLoss', eps=1e-3, naive_dice=False, use_sigmoid=False, loss_weight=1.0, loss_name='loss_dice'),
        ],
    ),

    auxiliary_head=dict(  # 轻量深监督，防止过强干扰主头
        type='FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=_NUM_CLASSES,
        norm_cfg=_NORM_CFG,
        align_corners=False,
        ignore_index=255,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2, loss_name='loss_ce_aux'),
            dict(type='DiceLoss', eps=1e-3, naive_dice=False, use_sigmoid=False, loss_weight=0.2, loss_name='loss_dice_aux'),
        ],
    ),

    train_cfg=dict(),
    test_cfg=dict(
        mode='slide',
        crop_size=_FINAL_SIZE,
        stride=(768, 768),   # 精度优先：重叠更大
    ),
)

# ========= Optim & Sched =========
optim_wrapper = dict(
    type='AmpOptimWrapper',  # 混精度提升稳定/吞吐
    optimizer=dict(type='AdamW', lr=6e-4, betas=(0.9, 0.999), weight_decay=0.01)
)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=0.9,
        begin=0,
        end=_MAX_EPOCH,
        by_epoch=True)
]

# ========= Train / Val / Test Loops =========
randomness = dict(seed=0)
train_cfg = dict(max_epochs=_MAX_EPOCH, type='EpochBasedTrainLoop', val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ========= Pipelines =========
# 统一长边，再做轻度仿射->裁剪；颜色增强保守，避免破坏舌色
_LONG_EDGE = 1536

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),

    dict(type='ResizeEdge', edge='long', scale=_LONG_EDGE, interpolation='lanczos', backend='cv2'),

    dict(type='RandomAffine',
         rotate_degree=8,
         scale=(0.94, 1.06),
         translate_ratio=(0.05, 0.05),
         border_val=0,  # 图像空洞填 0
         ),

    dict(type='RandomCrop', crop_size=_FINAL_SIZE, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),

    dict(type='ColorJitter',
         brightness=0.09, contrast=0.09, saturation=0.09, hue=[0.001, 0.009],
         backend='pillow'),

    dict(type='Albu',
         transforms=[
             dict(type='GaussNoise', var_limit=(5.0, 10.0), p=0.1)
         ],
         keymap={'img': 'image', 'gt_semantic_seg': 'mask'}),

    dict(type='GaussianBlur', ksize=3, sigma_min=0.1, sigma_max=0.3, prob=0.2),

    # 避免重复颜色扰动：去掉 PhotoMetricDistortion，ColorJitter 已覆盖
    dict(type='PackSegInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', edge='long', scale=_LONG_EDGE, keep_ratio=True, interpolation='lanczos', backend='cv2'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', edge='long', scale=_LONG_EDGE, keep_ratio=True, interpolation='lanczos', backend='cv2'),
    dict(type='PackSegInputs'),
]

# ========= Dataloaders =========
train_dataloader = dict(
    batch_size=8,              # 1024 crop 建议 8；显存OK可抬
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=_DATASET_TYPE,
        data_root=_DATA_ROOT,
        data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline,
        # reduce_zero_label=False  # 如你的 ZihaoDataset 用得到再加
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=_DATASET_TYPE,
        data_root=_DATA_ROOT,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=val_pipeline,
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=_DATASET_TYPE,
        data_root=_DATA_ROOT,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline,
    )
)

# ========= Metrics =========
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])

# ========= TTA =========
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[
            [  # multi-scale
                dict(type='Resize', keep_ratio=True, scale_factor=0.75),
                dict(type='Resize', keep_ratio=True, scale_factor=1.0),
                dict(type='Resize', keep_ratio=True, scale_factor=1.25),
            ],
            [  # flip
                dict(type='RandomFlip', direction='horizontal', prob=0.0),
                dict(type='RandomFlip', direction='horizontal', prob=1.0),
            ],
            [dict(type='PackSegInputs')]
        ])
]

# ========= Viz & Workdir =========
# 禁用可视化组件以避免MMCV C++扩展问题
# vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(type='SegLocalVisualizer', name='visualizer', vis_backends=[dict(type='LocalVisBackend')])
vis_backends = []
visualizer = dict(
    type='Visualizer',   # 来自 mmengine，不会触发 mmseg 的 mask_classification
    name='visualizer',
    vis_backends=[dict(type='LocalVisBackend')],
)
work_dir = './work_dirs/AN_UNet_1024_acc'
