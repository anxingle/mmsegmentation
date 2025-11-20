# 图像分割服务 (Image Segmentation Service)

基于 Flask 的 Web 应用程序，提供图像分割服务，使用训练好的 UNet 模型 (`work_dirs/AN_UNet_middle/epoch_5.pth`) 进行语义分割。

## 功能特点

- 📤 支持图片拖拽上传和点击选择
- 🔬 使用 `middle.pt` 模型进行图像分割
- 🎨 生成分割掩码 (Mask) 可视化
- 🌊 深绿色叠加效果显示分割区域
- 📱 响应式设计，支持移动端
- ⚡ 实时处理，即时反馈

## 系统要求

- Python 3.7+
- PyTorch
- OpenCV
- Flask

## 快速开始

### 1. 准备模型文件

确保以下模型文件存在：
- 训练好的 checkpoint: `work_dirs/AN_UNet_middle/epoch_5.pth`
- 配置文件: `an-Configs/AN_UNet_middle.py`

应用会直接使用 MMSegmentation 进行推理，无需额外转换 TorchScript。

### 2. 安装依赖

```bash
cd an-Configs/flask_js
pip install -r requirements.txt
```

### 3. 启动服务

```bash
# 方法1: 使用启动脚本
python run.py

# 方法2: 直接启动 Flask
python app.py
```

### 4. 访问服务

打开浏览器访问: http://localhost:5000

## API 接口

### 图片上传接口

**POST** `/upload`

**参数:**
- `file`: 图片文件 (支持 PNG, JPG, JPEG, BMP, TIFF)

**响应:**
```json
{
  "success": true,
  "original_image": "data:image/png;base64,...",
  "mask_image": "data:image/png;base64,...",
  "overlay_image": "data:image/png;base64,..."
}
```

### 健康检查接口

**GET** `/health`

**响应:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 项目结构

```
flask_js/
├── app.py              # Flask 主应用
├── run.py              # 启动脚本
├── requirements.txt    # Python 依赖
├── README.md          # 项目说明
├── templates/
│   └── index.html     # 前端页面
├── uploads/           # 临时上传目录
└── static/            # 静态资源目录
```

## 技术实现

### 后端 (Flask)

- **模型加载**: 使用 `torch.jit.load()` 加载 TorchScript 模型
- **图像预处理**:
  - 尺寸调整到 1024x1024
  - 归一化 (mean=[83.675, 156.28, 253.53], std=[78.395, 87.12, 5.375])
  - RGB 转换
- **推理**: 使用模型进行前向推理
- **后处理**:
  - Argmax 获取分割结果
  - 调整回原图尺寸
  - 生成掩码可视化
  - 创建深绿色叠加效果

### 前端 (HTML/CSS/JavaScript)

- **文件上传**: 支持拖拽和点击选择
- **实时反馈**: 加载状态、错误提示、成功消息
- **结果展示**: 原图、掩码、叠加效果并排显示
- **响应式设计**: 适配桌面和移动设备

## 配置参数

### 模型配置
- **输入尺寸**: 1024x1024
- **类别数**: 2 (背景 + 目标)
- **归一化参数**: mean=[83.675, 156.28, 253.53], std=[78.395, 87.12, 5.375]

### 叠加效果
- **叠加颜色**: 深绿色 (R:0, G:255, B:0)
- **透明度**: 0.6 (60%)
- **目标类别**: class 1

## 故障排除

### 1. 模型加载失败
```
错误: Error loading model
解决: 确保 middle.pt 文件存在于当前目录
```

### 2. 内存不足
```
错误: CUDA out of memory
解决: 重启服务，模型会自动使用 CPU
```

### 3. 文件上传失败
```
错误: File type not allowed
解决: 确保上传的文件格式为 PNG, JPG, JPEG, BMP, TIFF
```

### 4. 依赖包问题
```bash
# 安装所有依赖
pip install -r requirements.txt

# 如果遇到 OpenCV 问题
pip install opencv-python-headless
```

## 开发说明

### 自定义配置
可以在 `app.py` 中修改以下参数:
- `MAX_CONTENT_LENGTH`: 最大文件大小
- `ALLOWED_EXTENSIONS`: 支持的文件格式
- `target_size`: 模型输入尺寸
- `alpha`: 叠加透明度

### 扩展功能
- 支持批量处理
- 添加更多分割模型
- 实现用户认证
- 添加结果下载功能

## 许可证

本项目基于 MIT 许可证开源。