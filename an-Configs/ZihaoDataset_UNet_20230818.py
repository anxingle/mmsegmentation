_FINAL_SIZE = (
	1024,
	1024,
)
# XXX: 这里是分割的类别数
_NUM_CLASSES = 2
# XXX: 这里是数据集的根目录和类型
_DATA_ROOT = 'datasets/tongue_seg_v0/'
_DATASET_TYPE = 'ZihaoDataset'
_MAX_EPOCH = 500

_DATA_PREPROCESSOR = dict(
	bgr_to_rgb=True,
	mean=[123.675, 116.28, 103.53,],
	pad_val=0,
	seg_pad_val=255,
	size=(512, 1024,),
	std=[58.395, 57.12, 57.375,],
	type='SegDataPreProcessor'
)
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

# 制定了默认使用 `mmseg` 注册标，当配置中引用组件(模型、数据集、损失函数等)时，会优先从 `mmseg` 中查找已注册的模块
default_scope = 'mmseg'
env_cfg = dict(
	cudnn_benchmark=True,
	dist_cfg=dict(backend='nccl'),
	mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))

load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True)
# XXX: 这里是模型配置，使用UNet作为骨干网络
_NORM_CFG = dict(requires_grad=True, type='BN')
_PRETRRAINED_PATH = None
model = dict(
	auxiliary_head=dict(
		align_corners=False,
		channels=64,
		concat_input=False,
		dropout_ratio=0.1,
		in_channels=128,
		in_index=3,
		loss_decode=dict(
			loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
		norm_cfg=_NORM_CFG,
		num_classes=_NUM_CLASSES,
		num_convs=1,
		type='FCNHead'),
	backbone=dict(
		act_cfg=dict(type='ReLU'),
		base_channels=64,
		conv_cfg=None,
		dec_dilations=(1,1,1,1,),
		dec_num_convs=(2, 2, 2, 2,),
		downsamples=( True, True, True, True,),
		enc_dilations=(1, 1, 1, 1, 1,),
		enc_num_convs=( 2, 2, 2, 2, 2,),
		in_channels=3,
		norm_cfg=_NORM_CFG,
		norm_eval=False,
		num_stages=5,
		strides=( 1, 1, 1, 1, 1,),
		type='UNet',
		upsample_cfg=dict(type='InterpConv'),
		with_cp=False),
	data_preprocessor=_DATA_PREPROCESSOR,
	decode_head=dict(
		align_corners=False,
		channels=64,
		concat_input=False,
		dropout_ratio=0.1,
		in_channels=64,
		in_index=4,
		loss_decode=dict( loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
		norm_cfg=_NORM_CFG,
		num_classes=_NUM_CLASSES,
		num_convs=1,
		type='FCNHead'),
	# TODO: 预训练权重文件路径
	pretrained=_PRETRRAINED_PATH,
	test_cfg=dict(crop_size=256, mode='whole', stride=170),
	train_cfg=dict(),
	type='EncoderDecoder'
) # end of model


optim_wrapper = dict(
	clip_grad=None,
	optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
	type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)

param_scheduler = [dict(begin=0, by_epoch=True, end=_MAX_EPOCH, eta_min=0.0001, power=0.9, type='PolyLR'),]

randomness = dict(seed=0)
# TODO: 是否恢复训练
resume = False
train_cfg = dict(max_epochs=_MAX_EPOCH, type='EpochBasedTrainLoop', val_interval=5)
train_pipeline = [
	dict(type='LoadImageFromFile'), # 加载原始图像
	dict(type='LoadAnnotations'), # 加载对应的标注信息
	# 随机缩放数据增强 0.5-2.0 倍之间
	dict(type='RandomResize', keep_ratio=True, ratio_range=(0.5, 2.0,), scale=(2048, 1024,)),
	dict(cat_max_ratio=0.75, crop_size=_FINAL_SIZE, type='RandomCrop'),
	dict(prob=0.5, type='RandomFlip'),
	dict(type='PhotoMetricDistortion'), # 光照畸变数据增强
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
	sampler=dict(shuffle=True, type='InfiniteSampler')
) 

test_cfg = dict(type='TestLoop')
test_pipeline = [
	dict(type='LoadImageFromFile'),
	# TODO: 这里的 SCALE 如何调整呢？
	dict(keep_ratio=True, scale=(2048, 1024,), type='Resize'),
	dict(type='LoadAnnotations'),
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
work_dir = './work_dirs/ZihaoDataset-UNet'
