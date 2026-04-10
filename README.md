# CNN Character Recognition System

> **基于深度卷积神经网络的多类字符识别流水线**

本工程实现了一个基于 PyTorch 框架的高性能字符识别系统，支持对 **62 类字符（0-9, a-z, A-Z）** 的自动化识别与实时摄像头推理。系统在底层针对 Apple Silicon (M-series) 芯片进行了 Metal Performance Shaders (MPS) 计算后端的深度优化，同时完美兼容 CUDA 与 CPU 环境。

## 一、 神经网络架构机理 (Architectural Foundations)

本项目内置了三套阶梯式网络架构（`SimpleCNN`, `DetailedCNN`, `ResNet`），其核心数学与物理机理如下：

- **卷积层 (Spatial Feature Extraction)**

  通过离散卷积运算提取图像的局部边缘与纹理特征。其核心公式为：

  $$(f * g)(i, j) = \sum_m \sum_n f(m, n) \cdot g(i-m, j-n)$$

  其中 $f$ 为输入图像，$g$ 为卷积核（Kernel）。卷积运算具备局部感知（Local Receptive Fields）与权值共享（Weight Sharing）特性，极大减少了参数量并实现了平移不变性。

- **非线性激活 (ReLU)**

  引入非线性映射 $f(x) = \max(0, x)$ 打破线性变换的局限。ReLU 能够诱导神经元的稀疏激活性，并在反向传播中通过保持梯度恒定，有效缓解深层网络的梯度消失问题。

- **池化层 (Downsampling)**

  采用最大池化（Max Pooling）技术降低特征图的空间维度（例如从 64x64 压缩至 32x32），从而减少计算开销，并增强模型对于输入图像微小位移或形变的鲁棒性。

- **正则化防护 (Regularization)**

  - **Batch Normalization**: 强制对齐特征分布，缓解内部协变量偏移（Internal Covariate Shift），允许使用更大的学习率加速收敛。
  - **Dropout**: 在训练阶段随机“致盲”部分神经元，切断特征间的共适应性，强效防止模型死记硬背（过拟合）。

------

## 二、 项目目录结构 (Project Hierarchy)

本项目采用高度解耦的模块化设计，业务逻辑与底层工具严格分离：

Plaintext

```
Character_Recognition/
├── venv/                   # 📦 虚拟环境隔离目录 (Git Ignore)
├── data/                   # 🗂 数据管理层
│   ├── raw/                # 原始数据集 (Raw Assets)
│   └── processed/          # 归一化后的 64x64 灰度标准数据集
├── checkpoints/            # 💾 训练快照存档 (包含模型权重及优化器断点状态)
├── logs/                   # 📈 TensorBoard 训练监控日志
├── outputs/                # 📊 推理分析结果与深度诊断报告图表
├── src/                    # 🧠 核心算法包 (Algorithm Core)
│   ├── __init__.py         # 模块暴露接口与包管理实现
│   ├── model.py            # CNN 架构库 (SimpleCNN / DetailedCNN / ResNet)
│   ├── dataset.py          # 数据装载、全量内存缓存与增强流水线
│   ├── inference.py        # 静态推理引擎 (支持批量预测与 Top-K 分析)
│   └── utils.py            # 尺寸推演、可视化绘图与跨平台硬件调度
├── tools/                  # 🛠 辅助应用工具
│   ├── camera_app.py       # 实时 OpenCV 视频流监控应用 (支持 iPhone 互通摄像头)
│   ├── eval_matrix.py      # 模型期末考试：生成混淆矩阵与软肋排行榜
│   └── preprocess.py       # 增量式图像清洗与切分脚本
├── train.py                # 🚂 训练调度主入口 (Master Entry)
├── requirements.txt        # 📝 环境依赖 BOM 清单
└── README.md               # 📖 技术说明文档
```

------

## 三、 快速启动与部署 (Deployment Workflow)

### 1. 环境初始化

克隆本仓库后，请在虚拟环境中安装核心依赖：

Bash

```
python -m pip install -r requirements.txt
```

### 2. 数据清洗与预处理

执行图像重采样（64x64）、灰度化转换与严格的 Train/Val/Test 物理隔离。系统自带增量缓存机制，将自动跳过已处理素材：

Bash

```
python -m tools.preprocess
```

### 3. 模型训练

系统将自动探测并启用当前机器的最强算力（MPS/CUDA）。您可以通过 `--net` 参数自由切换底层架构模型：

Bash

```
# 启动训练任务 (默认使用 detailed 架构)
python train.py --net resnet

# 开启 TensorBoard 监控大屏 (另起一个终端运行)
tensorboard --logdir=logs
```

### 4. 实战推理模式

提供“静态诊断”与“实时视频流”两种实战模式：

Bash

```
# 模式 A：启动 iPhone / WebCam 实时视频流字符识别
python -m tools.camera_app --net resnet

# 模式 B：终端静态图像深度分析 (带 Top-K 概率图表输出)
python -m src.inference --image data/test.png --net resnet
```

------

## 四、 性能诊断与可视化 (Diagnostics)

通过运行 `python -m tools.eval_matrix --net resnet`，系统会在 `outputs/` 目录中生成深度诊断报告，协助您评估模型的真实泛化能力：

- 🎯 **全景混淆矩阵 (Confusion Matrix)**：量化跨类别分类误差，精准定位语义相近字符（如数字 `0` vs 大写字母 `O`，数字 `1` vs 小写字母 `l`）的区分度。
- 📉 **最易错软肋排行榜 (Weakness Top-K)**：自动提取矩阵中错误率最高的前 15 个字符对，并绘制直观的条形图，指导后续的针对性数据增强。
- 🔬 **特征图可视化 (Feature Maps)**：提取卷积层内部的激活值，像显微镜一样解析神经网络各阶段的注意力聚焦区域（*由 `utils.py` 提供支持*）。

------

## 五、 未来演进方向 (Future Directions)

随着计算机视觉（CV）领域的快速发展，本项目已预留良好的扩展接口，未来可作为以下前沿技术的基准测试床：

- **视觉 Transformer (ViT)**：引入自注意力机制（Self-Attention），突破卷积层局部感受野的先天限制，实现全局上下文建模。
- **多模态对齐 (CLIP)**：探索图像特征与自然语言文本的联合表征，迈向开放域（Open-vocabulary）的零样本（Zero-shot）字符识别。
- **模型量化与蒸馏 (TinyML)**：通过 INT8 极低精度量化或知识蒸馏（Knowledge Distillation）压缩网络参数，以适配算力严苛的嵌入式设备与边缘侧计算场景。
- **序列建模集成 (OCR)**：结合长短期记忆网络（LSTM）或 Transformer 解码器，将单字符识别升级为对连续手写单词/句子的端到端识别。
