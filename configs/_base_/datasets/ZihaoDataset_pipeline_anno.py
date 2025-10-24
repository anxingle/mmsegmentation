# 数据处理 pipeline 配置文件
# 用于舌象分割任务的数据预处理和数据加载配置

# ==================== 数据集基本配置 ====================
# 数据集类型，必须与 mmseg/datasets/ 中注册的类名对应
dataset_type = 'tongue_seg_v0'  # 舌象分割v0数据集类名
# 数据集根目录，相对于 mmsegmentation 主目录
data_root = 'datasets/tongue_seg_v0/'
# ==================== 图像预处理参数 ====================
# 输入模型的图像裁剪尺寸
# 要求：一般是 128 的倍数，这样对 GPU 计算友好
# 权衡：尺寸越小，显存开销越少，但可能损失细节；尺寸越大，精度可能更高但显存需求增加
crop_size = (1024, 1024)  # 高度 x 宽度，适合高分辨率医学图像
# ==================== 训练数据预处理 Pipeline ====================
# 训练时的数据增强和预处理流程，按顺序执行
train_pipeline = [
    # 第1步：从文件加载原始图像。读取图像文件到内存，支持多种格式（jpg, png等）
    dict(type='LoadImageFromFile'),
    # 第2步：加载标注数据。读取对应的分割标注图像（通常是单通道灰度图，像素值表示类别）
    dict(type='LoadAnnotations'),
    # 第3步：随机缩放增强。目的：增加尺度不变性，提高模型泛化能力
    dict(
        type='RandomResize',           # 随机缩放变换
        scale=(2048, 1024),           # 基础缩放尺寸（高度x宽度）
        ratio_range=(0.5, 2.0),       # 缩放比例范围：0.5倍到2.0倍
        keep_ratio=True               # 保持图像宽高比，避免形变
    ),
    # 第4步：随机裁剪。目的：固定输入尺寸，增加位置不变性
    dict(
        type='RandomCrop',            # 随机裁剪到固定尺寸
        crop_size=crop_size,          # 裁剪目标尺寸 (1024, 1024)
        cat_max_ratio=0.75            # 单个类别最大占比限制，避免裁剪区域过于单一
    ),
    # 第5步：随机水平翻转。目的：增加水平方向的不变性，数据增强
    dict(type='RandomFlip', prob=0.5),  # 50%概率进行水平翻转
    # 第6步：光度失真增强
    # 目的：模拟不同的光照条件，提高鲁棒性。包括：亮度、对比度、饱和度、色调的随机调整
    dict(type='PhotoMetricDistortion'),
    # 第7步：数据打包
    # 将图像和标注数据打包成模型输入格式，包括：
    # - 图像数据转换为张量
    # - 标注数据处理
    # - 添加元数据信息
    dict(type='PackSegInputs')
]
# ==================== 测试/验证数据预处理 Pipeline ====================
# 测试时不使用随机增强，保证结果可重现
test_pipeline = [
    # 第1步：从文件加载原始图像
    dict(type='LoadImageFromFile'),
    # 第2步：确定性缩放。目的：将图像缩放到合适尺寸供模型处理
    dict(
        type='Resize',                # 确定性缩放（非随机）
        scale=(2048, 1024),          # 目标尺寸
        keep_ratio=True              # 保持宽高比
    ),
    # 第3步：加载标注数据（用于验证）
    dict(type='LoadAnnotations'),
    # 第4步：数据打包
    dict(type='PackSegInputs')
]

# ==================== 测试时增强（TTA）配置 ====================
# TTA：Test Time Augmentation，测试时使用多个增强版本进行推理，然后融合结果
# 目的：提高推理精度，但会增加计算开销

# 多尺度缩放因子列表
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

# TTA 预处理流程
tta_pipeline = [
    # 第1步：加载图像，指定文件后端为磁盘读取
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),

    # 第2步：测试时增强变换
    dict(
        type='TestTimeAug',           # TTA主变换器
        transforms=[
            # 第一组变换：多尺度缩放
            # 对每个缩放因子都生成一个版本的图像
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios   # 列表推导式，生成6个不同尺度的Resize变换
            ],

            # 第二组变换：水平翻转
            # 对每个尺度版本，再生成原始和翻转两个版本
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),  # 不翻转
                dict(type='RandomFlip', prob=1., direction='horizontal')   # 必翻转
            ],

            # 第三组变换：加载标注（仅用于验证，测试时不需要）
            [dict(type='LoadAnnotations')],

            # 第四组变换：数据打包
            [dict(type='PackSegInputs')]
        ]
    )
]
# TTA 总共生成 6(尺度) × 2(翻转) = 12 个版本的预测结果

# ==================== 训练数据加载器配置 ====================
train_dataloader = dict(
    # 批次大小：每个GPU处理的样本数
    # 根据显存大小调整，显存小则减小此值
    batch_size=2,

    # 数据加载进程数：并行加载数据的子进程数量
    # 建议设置为CPU核心数的1/4到1/2，过多可能导致进程切换开销
    num_workers=2,

    # 持久化工作进程：True表示worker进程在epoch间不重启
    # 优点：避免重复初始化数据集，提高效率
    # 缺点：占用更多内存
    persistent_workers=True,

    # 数据采样器：控制数据读取顺序和策略
    sampler=dict(
        type='InfiniteSampler',       # 无限采样器：循环读取数据，适合长时间训练
        shuffle=True                  # 打乱数据顺序，提高训练随机性
    ),

    # 数据集配置
    dataset=dict(
        type=dataset_type,            # 使用前面定义的数据集类型
        data_root=data_root,          # 数据集根路径

        # 数据路径前缀配置
        data_prefix=dict(
            img_path='img_dir/train',     # 训练图像路径：{data_root}/img_dir/train
            seg_map_path='ann_dir/train'  # 训练标注路径：{data_root}/ann_dir/train
        ),

        pipeline=train_pipeline       # 使用训练预处理流程
    )
)

# ==================== 验证数据加载器配置 ====================
val_dataloader = dict(
    # 验证时批次大小通常设为1，便于逐样本处理和结果保存
    batch_size=1,

    # 验证时可使用更多worker，因为不需要梯度计算和权重更新
    num_workers=4,

    persistent_workers=True,          # 同样启用持久化worker

    # 验证采样器：顺序采样，确保验证结果的一致性和可重现性
    sampler=dict(
        type='DefaultSampler',         # 默认采样器
        shuffle=False                  # 不打乱顺序
    ),

    # 验证数据集配置
    dataset=dict(
        type=dataset_type,
        data_root=data_root,

        data_prefix=dict(
            img_path='img_dir/val',       # 验证图像路径
            seg_map_path='ann_dir/val'    # 验证标注路径
        ),

        pipeline=test_pipeline          # 使用测试预处理流程（无随机增强）
    )
)

# ==================== 测试数据加载器配置 ====================
# 测试时使用与验证相同的配置
test_dataloader = val_dataloader

# ==================== 验证评估器配置 ====================
# 定义验证时使用的评估指标
val_evaluator = dict(
    type='IoUMetric',                                 # IoU（交并比）指标评估器

    # 评估指标列表：
    # - mIoU: mean Intersection over Union，平均交并比，核心指标
    # - mDice: mean Dice coefficient，平均 Dice 系数，医学图像常用
    # - mFscore: mean F-score，平均 F 分数，综合考虑精确率和召回率
    iou_metrics=['mIoU', 'mDice', 'mFscore']
)

# ==================== 测试评估器配置 ====================
# 测试时使用与验证相同的评估配置
test_evaluator = val_evaluator