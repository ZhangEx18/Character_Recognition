"""
FastAPI 后端 - 神经网络字符识别平台

提供以下接口：
- POST /predict/ - 图片推理
- POST /train/start/ - 启动异步训练
- GET  /train/status/ - 获取训练状态
- GET  /train/metrics/ - 获取训练指标
- POST /train/stop/ - 停止训练
- GET  /outputs/ - 获取诊断图片
"""

#  uv run uvicorn backend:app --reload --port 8000

import io
import sys
import threading
import os
from datetime import datetime
from typing import Optional

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 导入模型和数据加载
from src import count_parameters, create_dataloaders, get_device
from src.models import create_model, MODEL_REGISTRY

# ============================================================
# 应用初始化
# ============================================================
app = FastAPI(title="神经网络字符识别 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8501", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 静态文件服务
# ============================================================
from starlette.responses import FileResponse

@app.get("/outputs/{filename}")
async def serve_output_file(filename: str):
    """提供 outputs 目录下的诊断图片"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "outputs", filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(file_path)

# ============================================================
# 训练状态管理
# ============================================================
training_state = {
    "status": "idle",  # idle | running | completed | stopped
    "net_type": "detailed",
    "epoch": 0,
    "epochs": 30,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "train_acc": 0.0,
    "val_acc": 0.0,
    "lr": 0.001,
    "best_acc": 0.0,
    "start_time": None,
    "history": [],  # 用于存储训练历史 [{epoch, train_loss, val_loss, train_acc, val_acc}, ...]
}

training_lock = threading.Lock()
training_thread: Optional[threading.Thread] = None


# ============================================================
# 预处理配置
# ============================================================
inference_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# 类别映射 (0-9: 数字, 10-35: 大写字母, 36-61: 小写字母)
def _get_class_name(idx: int) -> str:
    if idx < 10:
        return str(idx)
    elif idx < 36:
        return chr(ord('A') + idx - 10)
    else:
        return chr(ord('a') + idx - 36)

char_map = {i: _get_class_name(i) for i in range(62)}

# ============================================================
# 模型单例（延迟加载）
# ============================================================
_model_cache: dict = {}


def _get_model(net_type: str = "detailed") -> nn.Module:
    """获取推理用模型，始终使用 best_model.pth"""
    cache_key = f"inference_{net_type}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    model = create_model(net_type, num_classes=62)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(base_dir, "checkpoints", "best_model.pth")
    if os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"[Model] 已加载 checkpoint: {checkpoint_path}")
        except RuntimeError as e:
            print(f"[Model] checkpoint 与 {net_type} 不匹配，跳过加载，使用随机权重: {e}")

    model.eval()
    _model_cache[cache_key] = model
    return model


# ============================================================
# 推理接口
# ============================================================
@app.post("/predict/")
async def make_prediction(
    file: UploadFile = File(...),
    net_type: str = "detailed"
):
    """图片推理接口 - 使用 best_model.pth"""
    # 验证模型类型
    if net_type not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的模型类型: {net_type}。可用: {list(MODEL_REGISTRY.keys())}"
        )

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        tensor = inference_transform(image).unsqueeze(0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法解析图像: {e}")

    device = get_device()
    model = _get_model(net_type).to(device)
    tensor = tensor.to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = probs.max(1)

    class_idx = predicted.item()
    result = {
        "class": char_map[class_idx],
        "class_idx": class_idx,
        "confidence": round(confidence.item(), 4),
        "top_k": [
            {"class": char_map[idx], "confidence": round(probs[0][idx].item(), 4)}
            for idx in output[0].argsort(descending=True)[:5].tolist()
        ]
    }

    return {"filename": file.filename, "prediction": result}


# ============================================================
# 训练接口
# ============================================================
@app.post("/train/start/")
async def start_training(
    net_type: str = "detailed",
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    data_dir: str = "data/processed"
):
    """启动异步训练任务"""
    global training_thread

    if training_state["status"] == "running":
        raise HTTPException(status_code=400, detail="训练已在进行中")

    # 验证模型类型
    if net_type not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"不支持的模型类型: {net_type}。可用: {list(MODEL_REGISTRY.keys())}")

    # 重置训练状态
    with training_lock:
        training_state["status"] = "running"
        training_state["net_type"] = net_type
        training_state["epoch"] = 0
        training_state["epochs"] = epochs
        training_state["train_loss"] = 0.0
        training_state["val_loss"] = 0.0
        training_state["train_acc"] = 0.0
        training_state["val_acc"] = 0.0
        training_state["lr"] = learning_rate
        training_state["best_acc"] = 0.0
        training_state["start_time"] = datetime.now().isoformat()
        training_state["history"] = []

    # 获取基础目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, data_dir)
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    log_dir = os.path.join(base_dir, "logs")

    # 启动训练线程
    training_thread = threading.Thread(
        target=_train_model,
        args=(net_type, data_path, epochs, batch_size, learning_rate, checkpoint_dir, log_dir)
    )
    training_thread.daemon = True
    training_thread.start()

    return {"message": "训练已启动", "config": {
        "net_type": net_type,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    }}


def _train_model(
    net_type: str,
    data_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    checkpoint_dir: str,
    log_dir: str
):
    """后台训练函数 (在独立线程中运行)"""
    global training_state

    try:
        device = get_device()

        # 加载数据
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            image_size=64
        )

        # 创建模型
        model = create_model(net_type, num_classes=62).to(device)
        print(f"[训练] 模型: {net_type} | 参数量: {count_parameters(model):,}")

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_acc = 0.0

        for epoch in range(1, epochs + 1):
            # 检查是否被停止
            with training_lock:
                if training_state["status"] == "stopped":
                    break

            # 训练一个 epoch
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total

            # 验证
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()

            val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total

            # 更新状态
            with training_lock:
                training_state["epoch"] = epoch
                training_state["train_loss"] = train_loss
                training_state["val_loss"] = val_loss
                training_state["train_acc"] = train_acc
                training_state["val_acc"] = val_acc
                training_state["lr"] = optimizer.param_groups[0]['lr']
                training_state["history"].append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc
                })

            # 调度器
            scheduler.step(val_loss)

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                training_state["best_acc"] = best_acc
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))

        # 训练完成
        with training_lock:
            training_state["status"] = "completed"

    except Exception as e:
        with training_lock:
            training_state["status"] = "error"
            training_state["error"] = str(e)


@app.get("/train/status/")
async def get_training_status():
    """获取训练状态"""
    with training_lock:
        return {
            "status": training_state["status"],
            "net_type": training_state["net_type"],
            "epoch": training_state["epoch"],
            "epochs": training_state["epochs"],
            "best_acc": training_state["best_acc"],
            "start_time": training_state["start_time"]
        }


@app.get("/train/metrics/")
async def get_training_metrics():
    """获取训练指标"""
    with training_lock:
        return {
            "epoch": training_state["epoch"],
            "epochs": training_state["epochs"],
            "train_loss": training_state["train_loss"],
            "val_loss": training_state["val_loss"],
            "train_acc": training_state["train_acc"],
            "val_acc": training_state["val_acc"],
            "lr": training_state["lr"],
            "best_acc": training_state["best_acc"],
            "history": training_state["history"]
        }


@app.post("/train/stop/")
async def stop_training():
    """停止训练"""
    with training_lock:
        if training_state["status"] != "running":
            raise HTTPException(status_code=400, detail="当前没有训练在进行")
        training_state["status"] = "stopped"

    return {"message": "训练已停止"}


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
