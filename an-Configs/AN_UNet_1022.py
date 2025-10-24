# AN_UNet_1022.py
_FINAL_SIZE = (1024,1024)
# XXX: 这里是分割的类别数
_NUM_CLASSES = 2
# XXX: 这里是数据集的根目录和类型
_DATA_ROOT = 'datasets/tongue_seg_v0/'
_DATASET_TYPE = 'ZihaoDataset'
_MAX_EPOCH = 500

_DATA_PREPROCESSOR = dict(
	type='SegDataPreProcessor',
	bgr_to_rgb=True,
	mean=[123.675, 116.28, 103.53],
	std=[58.395, 57.12, 57.375],
	pad_val=0, # TODO: 这里是什么意思？
	seg_pad_val=255,  #  TODO: 这里 pad 的是什么？ 不应该设置为0吗？
)

# 制定了默认使用 `mmseg` 注册标，当配置中引用组件(模型、数据集、损失函数等)时，会优先从 `mmseg` 中查找已注册的模块
default_scope = 'mmseg'
env_cfg = dict(
	cudnn_benchmark=True,
	dist_cfg=dict(backend='nccl'),
	mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))

load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True)

# 定义在训练过程中的各种钩子(Hook), 在训练的不同阶段执行特定的功能
default_hooks = dict(
	checkpoint=dict(
		by_epoch=True, # 按照epoch保存
		interval=1, # 每10个epoch保存一次
		max_keep_ckpts=40, # 最多保存20个检查点
		save_best='mIoU', # 根据mIoU指标保存最佳模型
		type='CheckpointHook'),
	logger=dict(interval=1, log_metric_by_epoch=True, type='LoggerHook'),
	param_scheduler=dict(type='ParamSchedulerHook'), # 控制学习率等超参数的变换策略
	sampler_seed=dict(type='DistSamplerSeedHook'), # 确保分布式训练中数据采样的一致性
	timer=dict(type='IterTimerHook'), # 记录每次迭代的耗时，用于监控训练速度
	visualization=dict(type='SegVisualizationHook') #  可视化训练过程中的分割结果
)

# XXX: 这里是模型配置，使用UNet作为骨干网络
_NORM_CFG = dict(requires_grad=True, type='BN')
# _PRETRAINED_PATH = "checkpoint/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth" # XXX: 使用Cityscapes Dataset 预训练权重
_PRETRAINED_PATH = None  # TODO: 先不使用预训练权重
model = dict(
	type='EncoderDecoder',
	data_preprocessor=_DATA_PREPROCESSOR,
	backbone=dict(
		type='UNet',
		in_channels=3,
		base_channels=64,
		num_stages=5,
		strides=( 1, 1, 1, 1, 1),
		enc_num_convs=( 2, 2, 2, 2, 2),
		dec_num_convs=(2, 2, 2, 2,),
		enc_dilations=(1, 1, 1, 1, 1),
		dec_dilations=(1,1,1,1),
		downsamples=( True, True, True, True,),
		with_cp=False,
		norm_cfg=_NORM_CFG,
		act_cfg=dict(type='ReLU'),
		upsample_cfg=dict(type='InterpConv'),
		conv_cfg=None,
		norm_eval=False,
		),
	pretrained=_PRETRAINED_PATH,



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
		ignore_index=255,  # 忽略 _DATA_PREPROCESSOR 中的 seg_pad_val 设置
		loss_decode=[
			dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, use_sigmoid=False),
			dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0, use_sigmoid=False, naive_dice=False, eps=1e-3),
		],
		),

	auxiliary_head=dict(
		type='FCNHead',
		align_corners=False,
		channels=64,
		concat_input=False,
		dropout_ratio=0.1,
		in_channels=128,
		in_index=3,
		ignore_index=255,  # 忽略 _DATA_PREPROCESSOR 中的 seg_pad_val 设置
		loss_decode=[
			dict(type='CrossEntropyLoss', loss_name="loss_ce_aux", loss_weight=0.1, use_sigmoid=False),
			dict(type='DiceLoss', loss_name="loss_dice_aux", loss_weight=0.3, use_sigmoid=False, naive_dice=False, eps=1e-3),
			],
		norm_cfg=_NORM_CFG,
		num_classes=_NUM_CLASSES,
		num_convs=1),
	
	# test_cfg=dict(crop_size=256, mode='whole', stride=170),
	train_cfg=dict(),
	test_cfg=dict(mode='slide', crop_size=_FINAL_SIZE, stride=170),
) # end of model


optim_wrapper = dict(
	type='OptimWrapper',
	clip_grad=None,
	optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
	)

param_scheduler = [dict(begin=0, by_epoch=True, end=_MAX_EPOCH, eta_min=0.0001, power=0.9, type='PolyLR'),]

randomness = dict(seed=0)
# TODO: 是否恢复训练
resume = False
train_cfg = dict(max_epochs=_MAX_EPOCH, type='EpochBasedTrainLoop', val_interval=5)
# 增强版训练数据增强管道
train_pipeline = [
	dict(type='LoadImageFromFile'), # 加载原始图像
	dict(type='LoadAnnotations'), # 加载对应的标注信息
	# 1. 基础几何变换
	# 随机缩放数据增强 0.5-2.0 倍之间
	# dict(type='RandomResize', keep_ratio=True, ratio_range=(0.5, 2.0,), scale=(2048, 1024,)),
	# # 新增：随机旋转（舌体允许小幅旋转）
	# dict(
	# 	type='Rotate',
	# 	angle=8,  # 最大旋转角度8度（适合舌象分割）
	# 	prob=0.5,  # 50%的概率应用旋转
	# 	random_negative_prob=0.5,  # 随机方向旋转
	# 	pad_val=0,  # 旋转后空白区域填充黑色
	# 	interpolation='bilinear'
	# ),
	# # 新增：随机平移
	# dict(type='Translate', magnitude=0.04, prob=0.3, direction='horizontal', pad_val=0),
	# dict(type='Translate', magnitude=0.04, prob=0.3, direction='vertical', pad_val=0),
	# XXX: GPT 建议上述 `Rotate`, `Translate` 合并为
	dict(type='RandomAffine', rotate_degree=8, scale=(0.94, 1.06), translate_ratio=(0.05, 0.05), border_val=0),

	# 新增：边缘缩放预处理
	dict(
		type='ResizeEdge',
		scale=1280,  # 目标边缘长度
		edge='long',  # 长边缩放到1280
		interpolation='lanczos',  # 高质量插值
		backend='cv2'
	),

	# 新增：随机缩放裁剪（保守参数）
	dict(type='RandomCrop', crop_size=(1024, 1024), cat_max_ratio=0.75),

	# 2. 色彩变换（保守参数，适合舌象颜色判断）
	dict(
		type='ColorJitter',
		brightness=0.09, # 亮度扰动范围 91%~109% 之间随机调整
		contrast=0.09,   # 对比度
		saturation=0.09, # 饱和度（较小，避免影响舌象颜色）
		hue=[0.001, 0.009], # 色调偏移范围（很小）
		backend='pillow',
	),

	# 3. 噪声和模糊（保守参数）
	# 高斯噪声 - 使用 Albumentations TODO: 这里 keymap 是什么？
	dict(
		type='Albu',
		transforms=[dict(type='GaussNoise', std_range=(0.005, 0.01), p=0.5)],   # 高斯噪声（0.5%-1%的标准差）,
		keymap={'img': 'image', 'gt_semantic_seg': 'mask'}      # 把 mm 的 'img' 映射到 Albu 的 'image'
	),

	# 高斯模糊（模拟拍摄时的轻微失焦）
	# dict(
	# 	type='GaussianBlur',
	# 	magnitude_range=(0.1, 0.3),  # 较轻的模糊程度
	# 	magnitude_std='inf',
	# 	prob=0.5  # 50%概率应用模糊
	# ),
	# GPT 建议的高斯模糊
	dict(type='GaussianBlur', ksize=3, sigma_min=0.1, sigma_max=0.3, prob=0.6)

	# 4. 原有增强方法 TODO: RandomCrop 是什么？
	# dict(type='RandomCrop', cat_max_ratio=0.75, crop_size=_FINAL_SIZE),
	dict(type='RandomFlip', prob=0.5),
	dict(type='PhotoMetricDistortion'), # 光照畸变数据增强
	# dict(type='Pad', size=_FINAL_SIZE, pad_val=0, padding_mode='constant'), # TODO: 这里的 pad_val 和 seg_pad_val 需要确认
	dict(type='PackSegInputs'),
]

# XXX: 处理批级别操作，容器
train_dataloader = dict(
	batch_size=16,
	dataset=dict(
		type=_DATASET_TYPE,
		data_prefix=dict(
			img_path='img_dir/train', seg_map_path='ann_dir/train'),
		data_root=_DATA_ROOT,
		pipeline=train_pipeline,
	),
	num_workers=16,
	persistent_workers=True,
	sampler=dict(shuffle=True, type='DefaultSampler')
) 

test_cfg = dict(type='TestLoop')
test_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(type='LoadAnnotations'),
	# TODO: 不做 Resize, 使用原图
	# dict(type='Resize', keep_ratio=True, scale=(1024, 1024)), # 长边缩放到scale1, 短边按比例缩放
	dict(type='PackSegInputs'),
]

test_dataloader = dict(
	batch_size=1,
	dataset=dict(
		data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
		data_root=_DATA_ROOT,
		pipeline=test_pipeline,
		type='ZihaoDataset'),
	num_workers=4,
	persistent_workers=True,
	sampler=dict(shuffle=False, type='DefaultSampler')
)

test_evaluator = dict(iou_metrics=['mIoU','mDice','mFscore',], type='IoUMetric')

# 模型包装器：将普通的模型包装为支持TTA的形式
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
	dict(file_client_args=dict(backend='disk'), type='LoadImageFromFile'),
	dict(
		transforms=[
			[
				dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
				dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
				dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
				dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
			],
			[
				dict(direction='horizontal', prob=0.0, type='RandomFlip'),
				dict(direction='horizontal', prob=1.0, type='RandomFlip'),
			],
			[
				dict(type='LoadAnnotations'),
			],
			[
				dict(type='PackSegInputs'),
			],
		],
		type='TestTimeAug'),
]

val_cfg = dict(type='ValLoop')
val_dataloader = dict(
	batch_size=1,
	dataset=dict(
		data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
		data_root=_DATA_ROOT,
		pipeline=[
			dict(type='LoadImageFromFile'),
			dict(keep_ratio=True, scale=(
				2048,
				1024,
			), type='Resize'),
			dict(type='LoadAnnotations'),
			dict(type='PackSegInputs'),
		],
		type='ZihaoDataset'),
	num_workers=4,
	persistent_workers=True,
	sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
	iou_metrics=[
		'mIoU',
		'mDice',
		'mFscore',
	], type='IoUMetric')
vis_backends = [
	dict(type='LocalVisBackend'),
]
visualizer = dict(
	name='visualizer',
	type='SegLocalVisualizer',
	vis_backends=[
		dict(type='LocalVisBackend'),
	])
# 修改工作目录
work_dir = './work_dirs/AN_UNet_1022'
