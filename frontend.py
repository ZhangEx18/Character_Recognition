"""
Streamlit 前端 - 神经网络字符识别平台

参考 MotherDuck 设计风格：
- 配色: #fafafa 背景, #2d2d2d 文本, #3b82f6 强调色
- 字体: system-ui, -apple-system, sans-serif
- 卡片: 轻微边框 + 柔和阴影, 8px 圆角
- 布局: 清晰的网格系统, 充足的留白
"""
#  uv run streamlit run frontend.py --server.port 8501
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

# MotherDuck 风格 CSS
st.markdown("""
<style>
    /* ===== 基础变量 - MotherDuck 配色 ===== */
    :root {
        --bg-primary: #f8f9fa;
        --bg-secondary: #ffffff;
        --bg-tertiary: #e9ecef;
        --text-primary: #1a1a2e;
        --text-secondary: #495057;
        --text-muted: #6c757d;
        --border: #dee2e6;
        --border-hover: #ced4da;
        --accent: #4263eb;
        --accent-hover: #364fc7;
        --accent-soft: #edf2ff;
        --success: #2f9e44;
        --warning: #f08c00;
        --error: #e03131;
        --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.08);
        --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.12);
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 16px;
    }

    /* ===== 全局样式 ===== */
    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    }

    /* ===== 主标题 ===== */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.03em;
        margin-bottom: 0.35rem;
        line-height: 1.2;
    }

    .main-subtitle {
        font-size: 1rem;
        color: var(--text-secondary);
        font-weight: 400;
        margin-bottom: 2rem;
        letter-spacing: 0.01em;
    }

    /* ===== 卡片组件 ===== */
    .card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        box-shadow: var(--shadow-sm);
        transition: all 0.25s ease;
    }

    .card:hover {
        box-shadow: var(--shadow-md);
        border-color: var(--border-hover);
    }

    /* ===== Section 标题 ===== */
    .section-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: var(--text-primary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ===== Tab 样式 ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: transparent;
        border-bottom: 2px solid var(--border);
        padding-bottom: 0;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        border-bottom: 3px solid transparent;
        padding: 0.875rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        color: var(--text-muted);
        transition: all 0.2s ease;
        margin-bottom: -2px;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
        background: transparent;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--accent);
        background: transparent;
        border-bottom-color: var(--accent);
    }

    /* ===== 按钮样式 ===== */
    .stButton > button {
        background: var(--accent);
        color: white;
        border: none;
        border-radius: var(--radius-md);
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(66, 99, 235, 0.3);
    }

    .stButton > button:hover {
        background: var(--accent-hover);
        box-shadow: 0 4px 16px rgba(66, 99, 235, 0.4);
        transform: translateY(-2px);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    .stButton > button:disabled {
        background: var(--border);
        color: var(--text-muted);
        transform: none;
        box-shadow: none;
    }

    /* ===== 表单控件 ===== */
    .stSelectbox > div > div,
    .stSlider > div > div,
    .stNumberInput > div > div {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
    }

    .stSlider [data-baseweb="slider"] {
        color: var(--accent);
    }

    /* ===== 指标卡片 ===== */
    .metric-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.25rem 1.5rem;
        box-shadow: var(--shadow-sm);
        transition: all 0.25s ease;
    }

    .metric-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }

    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
        font-variant-numeric: tabular-nums;
    }

    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.5rem;
    }

    .metric-accent {
        color: var(--accent);
    }

    /* ===== 进度条 ===== */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--accent) 0%, #5c7cfa 100%);
        border-radius: 4px;
    }

    /* ===== 分隔线 ===== */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border), transparent);
        margin: 1.5rem 0;
    }

    /* ===== 信息提示 ===== */
    .stAlert {
        border-radius: var(--radius-md);
        border: none;
    }

    /* ===== 图表容器 ===== */
    .chart-container {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        box-shadow: var(--shadow-md);
    }

    /* ===== 侧边栏 ===== */
    .css-1d391kg {
        background: var(--bg-secondary);
    }

    /* ===== Radio 按钮 ===== */
    .stRadio > div {
        gap: 0.75rem;
    }

    .stRadio [data-baseweb="radio"] {
        padding: 0.75rem 1rem;
        border-radius: var(--radius-md);
        border: 2px solid var(--border);
        background: var(--bg-secondary);
        transition: all 0.2s ease;
        font-weight: 500;
    }

    .stRadio [data-baseweb="radio"]:hover {
        border-color: var(--accent);
        background: var(--accent-soft);
    }

    .stRadio [data-baseweb="radio"][aria-checked="true"] {
        border-color: var(--accent);
        background: var(--accent-soft);
        color: var(--accent);
    }

    /* ===== 上传区域 ===== */
    [data-testid="stFileUploader"] {
        background: var(--bg-secondary);
        border: 2px dashed var(--border);
        border-radius: var(--radius-lg);
        padding: 2rem;
        transition: all 0.2s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent);
        background: var(--accent-soft);
    }

    /* ===== 表格样式 ===== */
    table {
        width: 100%;
        font-size: 0.95rem;
    }

    td {
        padding: 0.75rem 0 !important;
        border-bottom: 1px solid var(--border) !important;
    }

    td:last-child {
        border-bottom: none !important;
    }

    /* Streamlit 原生元素覆盖 */
    .st-h1, .st-h2, .st-h3, .st-h4 {
        color: var(--text-primary);
        font-weight: 700;
    }

    p {
        color: var(--text-secondary);
    }

    /* 标签文字 */
    .stTabs label, .stRadio label {
        color: var(--text-primary);
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
        "params": "8.44M",
        "desc": "基础架构，适合快速测试"
    },
    "detailed": {
        "name": "DetailedCNN",
        "params": "4.32M",
        "desc": "BatchNorm + Dropout，稳健收敛"
    },
    "resnet": {
        "name": "ResNet",
        "params": "3.05M",
        "desc": "残差连接，适合复杂任务"
    }
}

# MotherDuck 风格图表配色
CHART_COLORS = {
    "train_loss": "#3b82f6",
    "val_loss": "#888888",
    "train_acc": "#22c55e",
    "val_acc": "#f59e0b"
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
        return {"error": "无法连接后端服务，请先启动后端"}
    except Exception as e:
        return {"error": str(e)}


def create_metrics_chart(history: list, height: int = 380) -> go.Figure:
    """创建训练指标图表 - MotherDuck 风格"""
    if not history:
        fig = go.Figure()
        fig.update_layout(
            height=height,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8f9fa",
            font={"color": "#495057", "family": "Inter, system-ui, sans-serif", "size": 12},
            xaxis=dict(showgrid=True, gridcolor="#dee2e6", color="#6c757d", showline=True, linewidth=1, linecolor="#dee2e6", tickfont=dict(size=11)),
            yaxis=dict(showgrid=True, gridcolor="#dee2e6", color="#6c757d", showline=True, linewidth=1, linecolor="#dee2e6", tickfont=dict(size=11)),
            margin=dict(l=65, r=40, t=50, b=60),
        )
        # 添加居中文字
        fig.add_annotation(
            text="📊 等待训练开始...",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=18, color="#adb5bd", family="Inter, system-ui, sans-serif"),
            xanchor="center",
            yanchor="middle"
        )
        # 添加副标题
        fig.add_annotation(
            text="选择模型并点击「开始训练」来查看实时指标",
            x=0.5, y=0.4,
            showarrow=False,
            font=dict(size=12, color="#868e96"),
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
        subplot_titles=["<b style='color:#1a1a2e;font-size:13px;font-family:Inter,sans-serif;'>📉 Loss 损失值</b>", "<b style='color:#1a1a2e;font-size:13px;font-family:Inter,sans-serif;'>📈 Accuracy 准确率 (%)</b>"],
        vertical_spacing=0.20
    )

    # Loss 子图 - 填充区域效果
    fig.add_trace(go.Scatter(
        x=epochs, y=train_loss,
        name="训练 Loss",
        line=dict(color="#4263eb", width=2.5, shape='spline'),
        mode="lines+markers",
        marker=dict(size=8, symbol="circle", line=dict(color="#ffffff", width=2)),
        fill='tonexty' if val_loss else None,
        fillcolor='rgba(66, 99, 235, 0.08)' if val_loss else None,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=epochs, y=val_loss,
        name="验证 Loss",
        line=dict(color="#868e96", width=2.5, dash="solid", shape='spline'),
        mode="lines+markers",
        marker=dict(size=8, symbol="diamond", line=dict(color="#ffffff", width=2)),
    ), row=1, col=1)

    # Accuracy 子图 - 填充区域效果
    fig.add_trace(go.Scatter(
        x=epochs, y=train_acc,
        name="训练准确率",
        line=dict(color="#2f9e44", width=2.5, shape='spline'),
        mode="lines+markers",
        marker=dict(size=8, symbol="circle", line=dict(color="#ffffff", width=2)),
        fill='tonexty' if val_acc else None,
        fillcolor='rgba(47, 158, 68, 0.08)' if val_acc else None,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=epochs, y=val_acc,
        name="验证准确率",
        line=dict(color="#f08c00", width=2.5, dash="solid", shape='spline'),
        mode="lines+markers",
        marker=dict(size=8, symbol="diamond", line=dict(color="#ffffff", width=2)),
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
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#dee2e6",
            borderwidth=1.5,
            borderradius=8,
            font=dict(family="Inter, system-ui, sans-serif", size=12, color="#495057")
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8f9fa",
        font={"color": "#1a1a2e", "family": "Inter, system-ui, sans-serif"},
        margin=dict(l=65, r=40, t=60, b=60),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#dee2e6",
            borderwidth=1.5,
            font=dict(family="Inter, system-ui, sans-serif", size=12)
        )
    )

    # X 轴配置
    fig.update_xaxes(
        title_text="<b style='color:#495057;font-size:12px;'>Epoch 轮次</b>",
        gridcolor="#dee2e6",
        color="#6c757d",
        showline=True,
        linewidth=1.5,
        linecolor="#dee2e6",
        tickfont=dict(size=11, family="Inter"),
        row=2, col=1
    )

    # Y 轴配置 - Loss
    fig.update_yaxes(
        title_text="<b style='color:#495057;font-size:12px;'>Loss</b>",
        gridcolor="#dee2e6",
        color="#6c757d",
        showline=True,
        linewidth=1.5,
        linecolor="#dee2e6",
        tickfont=dict(size=11, family="Inter"),
        row=1, col=1
    )

    # Y 轴配置 - Accuracy
    fig.update_yaxes(
        title_text="<b style='color:#495057;font-size:12px;'>Accuracy (%)</b>",
        gridcolor="#dee2e6",
        color="#6c757d",
        showline=True,
        linewidth=1.5,
        linecolor="#dee2e6",
        tickfont=dict(size=11, family="Inter"),
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
st.markdown('<p class="main-subtitle">支持多模型训练 · 实时可视化 · 62类字符识别 (0-9, a-z, A-Z)</p>', unsafe_allow_html=True)

# Tab 切换
tab_inference, tab_training = st.tabs(["🎯 推理模式", "🏋️ 训练模式"])

# ============================================================
# 推理模式
# ============================================================
with tab_inference:
    # 两列布局
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### 📤 上传图片")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "请选择要识别的图片",
            type=["jpg", "jpeg", "png", "bmp"],
            help="支持 JPG, PNG, BMP 格式"
        )

        if uploaded_file:
            st.image(uploaded_file, caption="待识别图片", width=320)
        else:
            st.info("请上传一张图片")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown("#### 📊 模型信息")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown("""
        <table style="width: 100%; font-size: 0.9rem;">
            <tr style="border-bottom: 1px solid #e5e5e5;">
                <td style="color: #888888; padding: 0.5rem 0;">当前模型</td>
                <td style="text-align: right; font-weight: 600; color: #2d2d2d;">DetailedCNN</td>
            </tr>
            <tr style="border-bottom: 1px solid #e5e5e5;">
                <td style="color: #888888; padding: 0.5rem 0;">参数量</td>
                <td style="text-align: right; font-weight: 600; color: #2d2d2d;">4.32M</td>
            </tr>
            <tr style="border-bottom: 1px solid #e5e5e5;">
                <td style="color: #888888; padding: 0.5rem 0;">输入尺寸</td>
                <td style="text-align: right; font-weight: 600; color: #2d2d2d;">64 × 64</td>
            </tr>
            <tr>
                <td style="color: #888888; padding: 0.5rem 0;">字符类别</td>
                <td style="text-align: right; font-weight: 600; color: #2d2d2d;">62 类</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("#### 🔤 支持的字符")
        st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
        st.markdown("**0-9 · a-z · A-Z**", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 推理按钮
    if uploaded_file:
        if st.button("🔮 开始识别", type="primary", use_container_width=True):
            with st.spinner("模型推理中..."):
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                result = call_api("POST", "/predict/", files=files)

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success("✅ 识别完成！")
                    prediction = result.get("prediction", {})
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("识别结果", prediction.get("class", "?"))
                    with col2:
                        st.metric("置信度", f"{prediction.get('confidence', 0)*100:.1f}%")

# ============================================================
# 训练模式
# ============================================================
with tab_training:
    # 获取训练状态
    status = call_api("GET", "/train/status/")
    current_status = status.get("status", "idle")

    # 顶部状态栏 - 4 个均等功能
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)

    status_labels = {
        "idle": "⏸️ 空闲",
        "running": "🔄 运行中",
        "completed": "✅ 已完成",
        "stopped": "⏹️ 已停止"
    }

    with s_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{status_labels.get(current_status, current_status)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">训练状态 · Epoch {status.get("epoch", 0)}/{status.get("epochs", 0)}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with s_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        best_acc = status.get("best_acc", 0.0)
        st.markdown(f'<div class="metric-value metric-accent">{"-" if not best_acc else f"{best_acc:.2f}%"}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">最佳准确率</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with s_col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        net_type = status.get("net_type", "detailed")
        st.markdown(f'<div class="metric-value" style="font-size: 1.2rem;">{MODELS.get(net_type, {}).get("name", net_type)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">当前模型</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with s_col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        start_time = status.get("start_time")
        elapsed = "-"
        if start_time:
            try:
                dt = datetime.fromisoformat(start_time)
                elapsed = (datetime.now() - dt).strftime("%H:%M:%S")
            except:
                pass
        st.markdown(f'<div class="metric-value" style="font-size: 1.2rem; font-variant-numeric: tabular-nums;">{elapsed}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">运行时长</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # 主体区域 - 配置 1/3，图表 2/3
    config_col, chart_col = st.columns([1, 2], gap="large")

    with config_col:
        st.markdown("##### ⚙️ 模型选择")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        # 使用单选按钮组选择模型
        selected = st.radio(
            "选择模型架构",
            options=list(MODELS.keys()),
            format_func=lambda x: f"{MODELS[x]['name']} ({MODELS[x]['params']})",
            index=list(MODELS.keys()).index(st.session_state.selected_model),
            label_visibility="collapsed",
            horizontal=True
        )

        st.session_state.selected_model = selected

        # 显示选中模型的详细信息
        model_info = MODELS[st.session_state.selected_model]
        st.markdown(f"""
        <div style="margin-top: 0.75rem; padding: 0.75rem; background: #eff6ff; border-radius: 8px; border-left: 3px solid #3b82f6;">
            <div style="font-weight: 600; color: #2d2d2d;">{model_info['name']}</div>
            <div style="color: #666666; font-size: 0.8rem; margin-top: 0.25rem;">{model_info['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("##### 📝 训练参数")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        epochs = st.slider("训练轮次 (Epochs)", 1, 100, 30)
        batch_size = st.select_slider("批次大小", options=[32, 64, 128, 256], value=128)
        learning_rate = st.number_input("学习率", value=0.001, format="%.4f", step=0.0001)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("##### 🎮 控制")
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
        st.markdown("##### 📈 训练曲线")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

        metrics = call_api("GET", "/train/metrics/")
        history = metrics.get("history", [])

        if current_status == "running" and history:
            st.session_state.training_history = history

        fig = create_metrics_chart(st.session_state.training_history or history, height=380)
        st.plotly_chart(fig, use_container_width=True)

        # 进度条
        if current_status == "running":
            epoch = metrics.get("epoch", 0)
            total_epochs = metrics.get("epochs", epochs)
            progress = epoch / total_epochs if total_epochs > 0 else 0
            st.progress(progress, text=f"训练进度: {epoch}/{total_epochs} ({progress*100:.1f}%)")

            # 实时指标
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
                <div style="margin-top: 1rem; padding: 1rem; background: #f0fdf4; border-radius: 8px; border-left: 3px solid #22c55e;">
                    <table style="width: 100%; font-size: 0.9rem;">
                        <tr style="border-bottom: 1px solid #e5e5e5;">
                            <td style="color: #666666;">最终训练准确率</td>
                            <td style="text-align: right; font-weight: 600; color: #22c55e;">{final.get('train_acc', 0):.2f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #e5e5e5;">
                            <td style="color: #666666;">最终验证准确率</td>
                            <td style="text-align: right; font-weight: 600; color: #f59e0b;">{final.get('val_acc', 0):.2f}%</td>
                        </tr>
                        <tr>
                            <td style="color: #666666;">最佳验证准确率</td>
                            <td style="text-align: right; font-weight: 700; color: #3b82f6;">{metrics.get('best_acc', 0):.2f}%</td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)

        elif current_status == "idle":
            st.info("👆 选择模型和参数后点击「开始训练」")

        st.markdown('</div>', unsafe_allow_html=True)

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
