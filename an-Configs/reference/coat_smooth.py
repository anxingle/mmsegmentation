_base_ = [
    '../../configs/_base_/models/efficientnet_v2/efficientnetv2_s.py',
    # '../../configs/_base_/schedules/imagenet_bs256.py',
    '../../configs/_base_/default_runtime.py',
]

# 模型设置 - 二分类任务
model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientNetV2', arch='xl'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=3,  # 滑(hua)/润(run)/燥(zao) 
        in_channels=1280,
        loss=dict(
            type='CrossEntropyLoss',
            # 类别权重：处理类别不平衡
            # ('hua', 'run', 'zao')数量为(42, 379, 124)
            class_weight=[0.02, 1.82, 0.01],  # [hua, run, zao]
            loss_weight=1.0),
        topk=(1,),  # 二分类只看top1准确率
    ))

# 数据预处理设置
data_preprocessor = dict(
    num_classes=3,  # 
    # RGB格式归一化参数（ImageNet标准）
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # 将图像从BGR转换为RGB
    to_rgb=True,
)

# 训练数据增强管道
# 注意：根据舌诊图像特点设计
train_pipeline = [
    dict(type='LoadImageFromFile'),
    
    # 1. 水平翻转（舌体左右对称，可以翻转）
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # 随机小角度旋转
    dict(
        type='Rotate',
        angle=12,  # 最大旋转角度12度
        prob=0.7,  # 70%的概率应用旋转
        random_negative_prob=0.5,  # 随机方向旋转正或负15度
        pad_val=0,  # 旋转后空白区域填充黑色，与Pad一致
        interpolation='bilinear'
    ),
    # 随机平移
    dict(type='Translate', magnitude=0.06, prob=0.3, direction='horizontal', pad_val=0),
    dict(type='Translate', magnitude=0.06, prob=0.3, direction='vertical', pad_val=0),
    # 随机缩放、裁剪
    dict(type='RandomResizedCrop', scale=384, crop_ratio_range=(0.95, 1.0), backend='pillow'),
    # Mixup - 暂时注释掉，避免错误
    # dict(type='Mixup', alpha=0.8, num_classes=3, probs=0.2),
    # XXX: 舌苔颜色 (灰、白、黄) 必须考虑 色彩、亮度、对比度、饱和度 的增强！ 
    # XXX: 慎用 ColorJitter， 会改变颜色！所以必须
    dict(
        type='ColorJitter',
        brightness=0.04, # 亮度扰动范围 96%~104% 之间随机调整
        contrast=0.04, # 对比度
        saturation=0.04, # 饱和度
        hue=[0.001, 0.01], # 色调偏移范围
        backend='pillow',
    ),
    # 轻微噪声 - 使用 Albumentations 高斯噪声
    dict(
        type='Albu',
        transforms=[
            dict(type='GaussNoise', std_range=(0.01, 0.02), p=0.9),   # 高斯噪声（1%-2%的标准差）
            # 可选：ISO 噪声（相机更像），二选一不要都开
            # dict(type='ISONoise', color_shift=(0.01, 0.03), intensity=(0.05, 0.15), p=0.0),
        ],
        keymap={'img': 'image'}      # 把 mm 的 'img' 映射到 Albu 的 'image'
    ),
    # 2. 轻度高斯模糊（模拟拍摄时的轻微失焦）
    dict(
        type='GaussianBlur',
        magnitude_range=(0.2, 0.5),  # 较轻的模糊程度
        magnitude_std='inf',
        prob=0.8  # 80%概率应用模糊
    ),
    
    # 3. 调整图像尺寸（保持长宽比，避免变形）
    dict(
        type='ResizeEdge',
        scale=384,  # 目标边缘长度
        edge='long',  # 长边缩放到384
        interpolation='lanczos',  # 高质量插值
        backend='cv2'
    ),
    
    # 4. 填充到正方形（避免裁切）
    dict(
        type='Pad',
        size=(384, 384),
        pad_val=0,
        padding_mode='constant'
    ),
    
    dict(type='PackInputs'),
]

# 测试/验证数据管道（无增强）
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=384, edge='long', backend='cv2'),
    dict(type='Pad', size=(384, 384), pad_val=0, padding_mode='constant'),
    dict(type='PackInputs'),
]

# 评估指标
val_evaluator = [
    dict(type='Accuracy', topk=(1,)),  # Top-1准确率
    dict(
        type='SingleLabelMetric',
        items=['precision', 'recall', 'f1-score'],
        average=None,  # None表示显示每个类别的指标
    )
]
test_evaluator = val_evaluator

# 数据加载器配置
train_dataloader = dict(
    batch_size=8,  # 根据显存调整
    num_workers=8,
    # metainfo=dict(classes=['gray', 'white', 'yellow]),
    dataset=dict(
        type="CustomDataset",
        data_root='/data-ssd/datasets/wet_smooth_dry_0910_bbox/',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    # metainfo=dict(classes=['gray', 'white', 'yellow]),
    dataset=dict(
        type="CustomDataset",
        data_root='/data-ssd/datasets/wet_smooth_dry_0910_bbox/',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    # metainfo=dict(classes=['gray', 'white', 'yellow]),
    dataset=dict(
        type="CustomDataset",
        data_root='/data-ssd/datasets/wet_smooth_dry_0910_bbox/',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# 优化器配置
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type='AdamW',  # 使用AdamW优化器，对于视觉任务效果更好
        lr=0.001,  # 初始学习率
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2)
)

# 学习率调度策略
param_scheduler = [
    # 预热阶段（前10个epoch）
    dict(
        type='LinearLR',
        start_factor=0.001,  # 从0.001倍开始
        by_epoch=True,
        begin=0,
        end=50,
        convert_to_iter_based=True,
    ),
    # 余弦退火阶段
    dict(
        type="CosineAnnealingLR",
        T_max=500,  # 总训练轮数
        eta_min=0.00001,  # 最小学习率
        by_epoch=True,
        begin=50,
        end=500,
    )
]

# 训练配置
train_cfg = dict(
    by_epoch=True, 
    max_epochs=500,  # 总训练轮数
    val_interval=5  # 每5个epoch验证一次
)
val_cfg = dict()
test_cfg = dict()

# 自动学习率缩放（根据batch size）
auto_scale_lr = dict(base_batch_size=256)

# 运行时配置
default_scope = 'mmpretrain'

# 默认钩子配置
default_hooks = dict(
    # 记录每次迭代的时间
    timer=dict(type='IterTimerHook'),
    # 每20次迭代打印日志
    logger=dict(type='LoggerHook', interval=20),
    # 启用参数调度器
    param_scheduler=dict(type='ParamSchedulerHook'),
    # 保存检查点
    checkpoint=dict(
        type='CheckpointHook', 
        interval=5,  # 每 5 个epoch保存
        max_keep_ckpts=100,  # 最多保留10个检查点
        save_best="auto"  # 自动保存最佳模型
    ),
    # 分布式采样器种子
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # 可视化钩子
    visualization=dict(type='VisualizationHook', enable=True),
)

# 自定义钩子：用于可视化训练过程中的图像
custom_hooks = [
    dict(
        type='VisualizationHook',
        enable=True,
        interval=20,  # 每20次迭代可视化一次
        show=False,
        draw_gt=True,
        draw_pred=True,
        out_dir='/data-ssd/logs/vis_teeth_normal',  # 可视化输出目录
    ),
]

# 环境配置
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# 可视化后端
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

# 日志级别
log_level = 'INFO'

# 加载预训练模型（EfficientNetV2-XL在ImageNet上的预训练权重）
# load_from = "/home/an/mmpretrain/works/efficientnetv2-s_in21k-pre-3rdparty_in1k_20221220-7a7c8475.pth"
load_from = "/home/an/mmpretrain/works/efficientnetv2-xl_in21k-pre-3rdparty_in1k_20221220-583ac18b.pth"
# 不恢复训练（从头开始）
resume = False

# 随机性配置
randomness = dict(seed=42, deterministic=False)  # 设置种子以保证可重复性

# 工作目录（保存日志和检查点）
work_dir = '/data-ssd/logs/coat_wet_smooth_dry_0910'
