"""
Streamlit 前端 - 神经网络字符识别平台

现代化 UI/UX 设计：
- 深色科技风格主题
- 流畅的动画和过渡
- Plotly 交互图表
"""

import streamlit as st
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="神经网络字符识别平台",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自定义 CSS - 深色科技风格
st.markdown("""
<style>
    /* 全局样式 */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }

    /* 主标题 */
    .main-header {
        font-family: 'Orbitron', 'Rajdhani', 'Segoe UI', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7c3aed, #00d4ff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    @keyframes shine {
        to { background-position: 200% center; }
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* 卡片样式 */
    .card {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .card-glow {
        position: relative;
        overflow: hidden;
    }

    .card-glow::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(0, 212, 255, 0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 0.5; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.1); }
    }

    /* 指标卡片 */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.2);
        border-color: rgba(0, 212, 255, 0.6);
    }

    .metric-value {
        font-family: 'JetBrains Mono', 'SF Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #00d4ff;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
    }

    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Tab 样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.5);
        padding: 8px;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 14px 28px;
        font-weight: 600;
        font-size: 1rem;
        color: #94a3b8;
        border: none;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 212, 255, 0.1);
        color: #00d4ff;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #7c3aed, #00d4ff);
        color: white;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
    }

    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #00d4ff);
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.5);
    }

    .stButton > button:disabled {
        background: #475569;
        box-shadow: none;
        transform: none;
    }

    /* 进度条 */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #7c3aed, #00d4ff, #7c3aed);
        background-size: 200% 100%;
        animation: gradient 2s linear infinite;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }

    /* 下拉框/选择器 */
    .stSelectbox > div > div, .stSlider > div > div {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }

    /* 分隔线 */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.3), transparent);
        margin: 1.5rem 0;
    }

    /* 模型选择卡片 */
    .model-option {
        background: rgba(30, 41, 59, 0.6);
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .model-option:hover {
        border-color: rgba(0, 212, 255, 0.5);
        background: rgba(0, 212, 255, 0.05);
    }

    .model-option.selected {
        border-color: #00d4ff;
        background: rgba(0, 212, 255, 0.1);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
    }

    /* 侧边栏 */
    .css-1d391kg {
        background: rgba(15, 23, 42, 0.9);
    }

    /* 滚动条 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(30, 41, 59, 0.5);
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #7c3aed, #00d4ff);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 配置
# ============================================================
BACKEND_URL = "http://127.0.0.1:8000"

MODELS = {
    "simple": {
        "name": "SimpleCNN",
        "params": "8.4M 参数",
        "desc": "基础架构 · 快速测试",
        "color": "#10b981"
    },
    "detailed": {
        "name": "DetailedCNN",
        "params": "1.2M 参数",
        "desc": "BatchNorm + Dropout · 稳健收敛",
        "color": "#00d4ff"
    },
    "resnet": {
        "name": "ResNet",
        "params": "1.8M 参数",
        "desc": "残差连接 · 深度网络",
        "color": "#7c3aed"
    }
}

# ============================================================
# 辅助函数
# ============================================================
def call_api(method: str, endpoint: str, files=None, **kwargs):
    """调用后端 API"""
    url = f"{BACKEND_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, params=kwargs, timeout=10)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, timeout=60)
            else:
                response = requests.post(url, json=kwargs, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "无法连接后端服务，请先启动后端: uvicorn backend:app --reload --port 8000"}
    except Exception as e:
        return {"error": str(e)}


def create_metrics_chart(history: list, height: int = 350) -> go.Figure:
    """创建训练指标图表 - 深色风格"""
    if not history:
        fig = go.Figure()
        fig.update_layout(
            height=height,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30,41,59,0.5)",
            font={"color": "#94a3b8", "family": "Helvetica Neue"},
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", color="#94a3b8"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", color="#94a3b8"),
        )
        fig.add_annotation(
            text="📊 等待训练开始...",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=18, color="#64748b"),
            xanchor="center",
            yanchor="middle"
        )
        return fig

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["<b>📉 Loss 曲线</b>", "<b>📈 Accuracy 曲线</b>"],
        vertical_spacing=0.15
    )

    # Loss
    fig.add_trace(go.Scatter(
        x=epochs, y=train_loss,
        name="训练 Loss",
        line=dict(color="#f97316", width=3),
        mode="lines+markers",
        marker=dict(size=8, symbol="circle")
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=epochs, y=val_loss,
        name="验证 Loss",
        line=dict(color="#ef4444", width=3, dash="dot"),
        mode="lines+markers",
        marker=dict(size=8, symbol="square")
    ), row=1, col=1)

    # Accuracy
    fig.add_trace(go.Scatter(
        x=epochs, y=train_acc,
        name="训练准确率",
        line=dict(color="#22c55e", width=3),
        mode="lines+markers",
        marker=dict(size=8, symbol="circle")
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=epochs, y=val_acc,
        name="验证准确率",
        line=dict(color="#00d4ff", width=3, dash="dot"),
        mode="lines+markers",
        marker=dict(size=8, symbol="square")
    ), row=2, col=1)

    fig.update_layout(
        height=height,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(30,41,59,0.8)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,41,59,0.3)",
        font={"color": "#e2e8f0", "family": "Helvetica Neue"},
        margin=dict(l=60, r=30, t=60, b=50),
    )

    fig.update_xaxes(
        title_text="训练轮次 (Epoch)",
        gridcolor="rgba(255,255,255,0.1)",
        color="#94a3b8",
        row=2, col=1
    )
    fig.update_yaxes(
        title_text="Loss 值",
        gridcolor="rgba(255,255,255,0.1)",
        color="#94a3b8",
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="准确率 (%)",
        gridcolor="rgba(255,255,255,0.1)",
        color="#94a3b8",
        row=2, col=1
    )

    return fig


# ============================================================
# 状态初始化
# ============================================================
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "detailed"

if "training_history" not in st.session_state:
    st.session_state.training_history = []


# ============================================================
# 主界面
# ============================================================
st.markdown('<h1 class="main-header">🔤 神经网络字符识别平台</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">支持多模型训练 · 实时可视化 · 62类字符识别 (0-9, a-z, A-Z)</p>', unsafe_allow_html=True)

# Tab 切换
tab_inference, tab_training = st.tabs(["🎯 推理模式", "🏋️ 训练模式"])

# ============================================================
# 推理模式
# ============================================================
with tab_inference:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📤 上传图片")
        uploaded_file = st.file_uploader(
            "请选择要识别的图片",
            type=["jpg", "jpeg", "png", "bmp"],
            help="支持 JPG, PNG, BMP 格式"
        )

        if uploaded_file:
            st.image(uploaded_file, caption="待识别图片", width=380)

            if st.button("🔮 开始识别", type="primary", use_container_width=True):
                with st.spinner("🔄 模型推理中，请稍候..."):
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                    }
                    result = call_api("POST", "/predict/", files=files)

                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success("✅ 识别完成！")
                        prediction = result.get("prediction", {})
                        st.metric(
                            label="识别结果",
                            value=prediction.get("class", "?"),
                            delta=f"置信度: {prediction.get('confidence', 0)*100:.1f}%"
                        )
        else:
            st.info("👆 请上传一张图片开始识别")

    with col2:
        st.markdown("### 📊 模型信息")
        st.markdown("""
        <div class="card">
            <h4 style="color: #00d4ff; margin-top: 0;">当前模型</h4>
            <p style="font-size: 1.4rem; color: #ffffff; margin: 12px 0;">
                <strong>DetailedCNN</strong>
            </p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px;">
                <div style="background: rgba(0,212,255,0.1); padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="color: #00d4ff; font-size: 1.2rem;">1.2M</div>
                    <div style="color: #94a3b8; font-size: 0.8rem;">参数量</div>
                </div>
                <div style="background: rgba(124,58,237,0.1); padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="color: #7c3aed; font-size: 1.2rem;">64×64</div>
                    <div style="color: #94a3b8; font-size: 0.8rem;">输入尺寸</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔤 支持的字符")
        chars = "0-9 · a-z · A-Z"
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <p style="font-size: 1.5rem; color: #00d4ff; letter-spacing: 4px;">{chars}</p>
            <p style="color: #94a3b8;">共 62 个类别</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# 训练模式
# ============================================================
with tab_training:
    # 顶部状态卡片
    status = call_api("GET", "/train/status/")
    current_status = status.get("status", "idle")

    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    status_map = {
        "idle": "⏸️ 空闲",
        "running": "🔄 训练中",
        "completed": "✅ 已完成",
        "stopped": "⏹️ 已停止"
    }

    with s_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{status_map.get(current_status, current_status)}</div>
            <div class="metric-label">训练状态</div>
            <div style="color: #64748b; font-size: 0.8rem; margin-top: 4px;">
                Epoch {status.get('epoch', 0)}/{status.get('epochs', 0)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with s_col2:
        best_acc = status.get("best_acc", 0.0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{'-' if not best_acc else f'{best_acc:.2f}%'}</div>
            <div class="metric-label">最佳准确率</div>
        </div>
        """, unsafe_allow_html=True)

    with s_col3:
        net_type = status.get("net_type", "detailed")
        model_info = MODELS.get(net_type, {})
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size: 1.5rem;">{model_info.get('name', net_type)}</div>
            <div class="metric-label">当前模型</div>
        </div>
        """, unsafe_allow_html=True)

    with s_col4:
        start_time = status.get("start_time")
        elapsed = "-"
        if start_time:
            try:
                dt = datetime.fromisoformat(start_time)
                elapsed = (datetime.now() - dt).strftime("%H:%M:%S")
            except:
                pass
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size: 1.3rem;">{elapsed}</div>
            <div class="metric-label">运行时长</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # 配置和图表区域
    config_col, chart_col = st.columns([1, 2], gap="large")

    with config_col:
        st.markdown("### ⚙️ 模型选择")

        for model_key, model_info in MODELS.items():
            is_selected = st.session_state.selected_model == model_key
            card_class = "model-option selected" if is_selected else "model-option"

            if st.button(
                f"**{model_info['name']}**\n{model_info['params']}\n_{model_info['desc']}_",
                key=f"model_{model_key}"
            ):
                st.session_state.selected_model = model_key
                st.rerun()

        st.markdown("### 📝 训练参数")
        epochs = st.slider("训练轮次 (Epochs)", 1, 100, 30)
        batch_size = st.select_slider("批次大小 (Batch Size)", options=[32, 64, 128, 256], value=128)
        learning_rate = st.number_input("学习率 (Learning Rate)", value=0.001, format="%.4f")

        st.markdown("### 🎮 控制")
        c1, c2 = st.columns(2)
        with c1:
            start_btn = st.button("▶️ 开始训练", type="primary", use_container_width=True, disabled=current_status == "running")
        with c2:
            stop_btn = st.button("⏹️ 停止", use_container_width=True, disabled=current_status != "running")

        if start_btn and current_status != "running":
            result = call_api("POST", "/train/start/",
                net_type=st.session_state.selected_model,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            if "error" in result:
                st.error(result["error"])
            else:
                st.rerun()

        if stop_btn and current_status == "running":
            result = call_api("POST", "/train/stop/")
            if "error" not in result:
                st.rerun()

    with chart_col:
        st.markdown("### 📈 训练曲线")

        metrics = call_api("GET", "/train/metrics/")
        history = metrics.get("history", [])

        if current_status == "running" and history:
            st.session_state.training_history = history

        fig = create_metrics_chart(st.session_state.training_history or history, height=400)
        st.plotly_chart(fig, use_container_width=True)

        # 进度条
        if current_status == "running":
            epoch = metrics.get("epoch", 0)
            total_epochs = metrics.get("epochs", epochs)
            progress = epoch / total_epochs if total_epochs > 0 else 0
            st.progress(progress, text=f"🔄 训练进度: {epoch}/{total_epochs} ({progress*100:.1f}%)")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("训练 Loss", f"{metrics.get('train_loss', 0):.4f}")
            with m2:
                st.metric("验证 Loss", f"{metrics.get('val_loss', 0):.4f}")
            with m3:
                st.metric("训练准确率", f"{metrics.get('train_acc', 0):.2f}%")
            with m4:
                st.metric("验证准确率", f"{metrics.get('val_acc', 0):.2f}%")

            time.sleep(2)
            st.rerun()

        elif current_status == "completed":
            st.success("🎉 训练已完成！")
            if history:
                final = history[-1]
                st.markdown(f"""
                <div class="card">
                    <h4 style="color: #00d4ff; margin-top: 0;">📋 训练报告</h4>
                    <table style="width: 100%; color: #e2e8f0;">
                        <tr>
                            <td>最终训练准确率</td>
                            <td style="text-align: right; color: #22c55e; font-weight: bold;">{final.get('train_acc', 0):.2f}%</td>
                        </tr>
                        <tr>
                            <td>最终验证准确率</td>
                            <td style="text-align: right; color: #00d4ff; font-weight: bold;">{final.get('val_acc', 0):.2f}%</td>
                        </tr>
                        <tr>
                            <td>最佳验证准确率</td>
                            <td style="text-align: right; color: #f97316; font-weight: bold;">{metrics.get('best_acc', 0):.2f}%</td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

        elif current_status == "idle":
            st.info("👆 选择模型和参数后点击「开始训练」")

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.markdown("### ℹ️ 关于")
    st.markdown("""
    **神经网络字符识别平台**

    🔹 支持三种 CNN 架构：
    - SimpleCNN - 基础架构
    - DetailedCNN - 生产级
    - ResNet - 深度网络

    🔹 技术栈：
    - PyTorch 深度学习
    - FastAPI 后端服务
    - Streamlit 前端界面

    ---

    **启动后端：**
    ```bash
    uvicorn backend:app --reload --port 8000
    ```
    """)
