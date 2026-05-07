# Character Recognition - 多架构神经网络字符识别平台

> **基于 PyTorch 的全栈字符识别系统，支持 8 种神经网络架构**

本项目整合多个神经网络实现，统一适配 **62 类字符（0-9, a-z, A-Z）** 识别任务，提供 FastAPI 后端 + Streamlit 前端的全栈体验。针对 Apple Silicon (MPS)、CUDA 和 CPU 进行了跨平台优化。

---

## 支持的神经网络架构

| 架构 | 类型 | 特点 |
|------|------|------|
| **SimpleCNN** | 卷积网络 | 2层卷积 + 2层FC，参数量最小，快速验证首选 |
| **DetailedCNN** | 卷积网络 | 3层卷积 + BN + Dropout，生产环境默认推荐 |
| **ResNet** | 残差网络 | 4-stage 轻量残差，32→256通道，深层特征提取 |
| **SE-ResNet** | 残差网络 | ResNet + SE-Block 通道注意力，增强难例区分 |
| **MLPNet** | 全连接网络 | 纯FC无卷积，4096→512→256→62，baseline对比 |
| **CifarCNN** | 卷积网络 | CIFAR风格双块池化，适合小数据集快速实验 |
| **ResNet-18** | 残差网络 | 标准ResNet-18，4-stage 64→512通道 |
| **ResNet-20** | 残差网络 | 轻量3-stage，16→64通道，参数量极小 |

---

## 项目结构

```
Character_Recognition/
├── backend.py               # FastAPI 后端（推理 + 训练 API）
├── frontend.py              # 前端 SPA（Neural Dark 主题）
├── train.py                 # 训练入口（支持所有架构）
├── requirements.txt         # 依赖清单
├── src/                     # 核心源码
│   ├── __init__.py
│   ├── models/              # 神经网络架构库
│   │   ├── __init__.py      # 模型注册表 + create_model 工厂
│   │   ├── simple_cnn.py
│   │   ├── detailed_cnn.py
│   │   ├── resnet.py        # ResNet + ResNet18 + ResNet20
│   │   ├── seresnet.py      # SE-ResNet + SE-Block
│   │   ├── mlp_net.py       # 全连接网络
│   │   ├── cifar_cnn.py     # CIFAR风格CNN
│   │   └── base.py          # 通用工具函数
│   ├── model.py             # FocalLoss（保留）
│   ├── dataset.py           # 数据加载与增强
│   ├── inference.py         # 推理引擎（Predictor）
│   └── utils.py             # 可视化 + 硬件调度
├── tools/                   # 辅助工具
│   ├── camera_app.py        # 实时摄像头推理
│   ├── eval_matrix.py       # 混淆矩阵与弱点分析
│   └── preprocess.py        # 数据预处理
├── data/                    # 数据集（train/val/test）
├── checkpoints/             # 模型权重存档
├── logs/                    # TensorBoard 日志
└── outputs/                 # 推理结果与诊断图表
```

---

## 快速启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务

```bash
# 终端 1：启动 FastAPI 后端（端口 8000）
python backend.py

# 终端 2：启动前端（端口 8501）
python frontend.py
# 访问 http://127.0.0.1:8501/
```

### 3. 模型训练

```bash
# 默认使用 DetailedCNN
python train.py

# 指定其他架构
python train.py --net resnet18
python train.py --net seresnet
python train.py --net mlp

# 查看所有可用架构
python train.py --net  # 会提示可用选项

# TensorBoard 监控
tensorboard --logdir=logs
```

### 4. 推理

```bash
# 命令行推理
python -m src.inference --image data/test.png --net resnet18

# 实时摄像头
python -m tools.camera_app --net detailed
```

---

## 技术特性

- **8 种神经网络架构**：从简单 MLP 到深层 ResNet，覆盖不同复杂度需求
- **统一接口**：所有模型通过 `create_model(name, num_classes=62)` 一键创建
- **全栈架构**：FastAPI 后端 + 现代化前端，支持训练监控与实时推理
- **数据增强**：RandomRotation、Affine、Perspective、ElasticTransform、RandomErasing
- **硬件自适应**：自动检测 MPS / CUDA / CPU
- **训练可视化**：TensorBoard + 前端实时曲线

---

## 未来方向

- Vision Transformer (ViT) 架构支持
- 模型量化与知识蒸馏
- ONNX 导出与边缘部署
- 连续文本 OCR 序列建模
