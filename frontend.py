"""
单文件 Web 前端 - 神经网络字符识别平台

美学方向：Neural Dark
- 深色背景配合霓虹蓝/绿点缀
- 玻璃拟态卡片 + 微妙光晕
- 神经网络节点动画背景
- 训练曲线实时渲染
- 精致动画过渡

启动方式:
    uv run python frontend.py
"""

import os
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler


class SPAHandler(SimpleHTTPRequestHandler):
    """Serve the CNN frontend at /, everything else returns 404."""

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.path = '/.tmp/frontend.html'
        return SimpleHTTPRequestHandler.do_GET(self)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        SimpleHTTPRequestHandler.end_headers(self)


HTML_CONTENT = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CNN 字符识别平台</title>
<style>
  :root {
    --bg-void: #070b14;
    --bg-base: #0d1117;
    --bg-card: rgba(22, 27, 34, 0.85);
    --border: rgba(48, 54, 61, 0.8);
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-muted: #484f58;
    --accent-blue: #38bdf8;
    --accent-cyan: #22d3ee;
    --accent-green: #4ade80;
    --accent-amber: #fbbf24;
    --accent-red: #f87171;
    --glow-blue: 0 0 20px rgba(56, 189, 248, 0.15);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 18px;
    --font-mono: 'JetBrains Mono', 'Fira Code', ui-monospace, monospace;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: var(--font-sans);
    background: var(--bg-void);
    color: var(--text-primary);
    min-height: 100vh;
    overflow-x: hidden;
    line-height: 1.6;
  }
  #neural-bg { position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden; }
  #neural-bg canvas { width: 100%; height: 100%; opacity: 0.3; }
  .app { position: relative; z-index: 1; min-height: 100vh; display: flex; flex-direction: column; }
  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 2rem;
    border-bottom: 1px solid var(--border);
    backdrop-filter: blur(12px);
    background: rgba(13, 17, 23, 0.7);
    position: sticky; top: 0; z-index: 100;
  }
  .header-brand { display: flex; align-items: center; gap: 0.75rem; }
  .header-title {
    font-size: 1.1rem; font-weight: 700; letter-spacing: -0.02em;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  }
  .header-badge {
    font-size: 0.65rem; font-family: var(--font-mono);
    background: rgba(56, 189, 248, 0.1); color: var(--accent-blue);
    border: 1px solid rgba(56, 189, 248, 0.2);
    padding: 0.15rem 0.5rem; border-radius: 20px; letter-spacing: 0.05em;
  }
  .header-status { display: flex; align-items: center; gap: 0.5rem; font-size: 0.8rem; font-family: var(--font-mono); color: var(--text-secondary); }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-muted); transition: background 0.3s; }
  .status-dot.connected { background: var(--accent-green); box-shadow: 0 0 6px var(--accent-green); }
  .status-dot.training { background: var(--accent-amber); box-shadow: 0 0 6px var(--accent-amber); animation: pulse 1.2s ease-in-out infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
  .main { flex: 1; max-width: 1280px; margin: 0 auto; width: 100%; padding: 2rem; }
  .nav-tabs {
    display: flex; gap: 0.25rem; margin-bottom: 2rem;
    background: rgba(22, 27, 34, 0.5);
    border: 1px solid var(--border); border-radius: var(--radius-lg);
    padding: 0.35rem; width: fit-content;
  }
  .nav-tab {
    padding: 0.6rem 1.5rem; border-radius: var(--radius-md);
    font-size: 0.875rem; font-weight: 600; cursor: pointer;
    border: none; background: transparent; color: var(--text-secondary);
    transition: all 0.2s ease; display: flex; align-items: center; gap: 0.5rem;
  }
  .nav-tab:hover { color: var(--text-primary); background: rgba(56, 189, 248, 0.06); }
  .nav-tab.active { background: rgba(56, 189, 248, 0.12); color: var(--accent-blue); box-shadow: var(--glow-blue); }
  .panel { display: none; }
  .panel.active { display: block; }
  .card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 1.5rem;
    backdrop-filter: blur(16px); transition: border-color 0.3s, box-shadow 0.3s;
  }
  .card:hover { border-color: rgba(56, 189, 248, 0.2); }
  .card-title {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 1.25rem;
    display: flex; align-items: center; gap: 0.5rem;
  }
  .inference-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
  @media (max-width: 768px) { .inference-grid { grid-template-columns: 1fr; } }
  .drop-zone {
    border: 2px dashed var(--border); border-radius: var(--radius-lg);
    padding: 3rem 2rem; text-align: center; cursor: pointer;
    transition: all 0.25s ease; position: relative; overflow: hidden;
  }
  .drop-zone::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at center, rgba(56, 189, 248, 0.05), transparent 70%);
    opacity: 0; transition: opacity 0.3s;
  }
  .drop-zone:hover, .drop-zone.drag-over {
    border-color: var(--accent-blue); box-shadow: var(--glow-blue), inset 0 0 40px rgba(56, 189, 248, 0.03);
  }
  .drop-zone:hover::before, .drop-zone.drag-over::before { opacity: 1; }
  .drop-zone-icon { font-size: 2.5rem; margin-bottom: 1rem; filter: grayscale(0.3); transition: filter 0.3s, transform 0.3s; }
  .drop-zone:hover .drop-zone-icon, .drop-zone.drag-over .drop-zone-icon { filter: grayscale(0); transform: scale(1.1); }
  .drop-zone-text { color: var(--text-secondary); font-size: 0.9rem; }
  .drop-zone-text strong { color: var(--accent-blue); }
  .drop-zone-hint { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem; font-family: var(--font-mono); }
  #preview-img { max-width: 100%; max-height: 260px; border-radius: var(--radius-md); display: none; margin: 1rem auto; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
  .model-selector { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; margin-bottom: 1.25rem; }
  .model-option {
    border: 1px solid var(--border); border-radius: var(--radius-md);
    padding: 0.85rem 0.75rem; cursor: pointer;
    transition: all 0.2s ease; background: rgba(13, 17, 23, 0.4); text-align: center;
  }
  .model-option:hover { border-color: var(--accent-blue); background: rgba(56, 189, 248, 0.05); }
  .model-option.selected { border-color: var(--accent-blue); background: rgba(56, 189, 248, 0.1); box-shadow: var(--glow-blue); }
  .model-option-name { font-size: 0.85rem; font-weight: 700; color: var(--text-primary); }
  .model-option-params { font-size: 0.7rem; color: var(--text-muted); font-family: var(--font-mono); margin-top: 0.25rem; }
  .model-selector { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; }
  .model-option { min-width: 0; }
  .form-select {
    width: 100%; padding: 0.6rem 0.85rem;
    background: rgba(13, 17, 23, 0.6); color: var(--text-primary);
    border: 1px solid var(--border); border-radius: var(--radius-md);
    font-size: 0.875rem; font-family: var(--font-sans); font-weight: 600;
    cursor: pointer; appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%238b949e' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: right 0.75rem center;
    padding-right: 2.2rem; transition: border-color 0.2s;
  }
  .form-select:focus { outline: none; border-color: var(--accent-blue); box-shadow: 0 0 0 2px rgba(56,189,248,0.15); }
  .form-select option { background: #0d1117; color: var(--text-primary); }
  .btn {
    display: inline-flex; align-items: center; justify-content: center; gap: 0.5rem;
    padding: 0.75rem 1.75rem; border-radius: var(--radius-md);
    font-size: 0.875rem; font-weight: 700; cursor: pointer; border: none;
    transition: all 0.2s ease; font-family: var(--font-sans); letter-spacing: 0.01em;
  }
  .btn-primary { background: linear-gradient(135deg, #0369a1, #0ea5e9); color: white; box-shadow: 0 4px 16px rgba(14, 165, 233, 0.3); }
  .btn-primary:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 6px 24px rgba(14, 165, 233, 0.4); }
  .btn-primary:active:not(:disabled) { transform: translateY(0); }
  .btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-secondary { background: rgba(48, 54, 61, 0.6); color: var(--text-secondary); border: 1px solid var(--border); }
  .btn-secondary:hover:not(:disabled) { border-color: var(--text-muted); color: var(--text-primary); }
  .btn-danger { background: rgba(248, 113, 113, 0.1); color: var(--accent-red); border: 1px solid rgba(248, 113, 113, 0.3); }
  .btn-danger:hover:not(:disabled) { background: rgba(248, 113, 113, 0.2); }
  .btn-full { width: 100%; }
  .result-panel { display: none; animation: fadeIn 0.4s ease; }
  .result-panel.show { display: block; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
  .result-char {
    font-size: 5rem; font-weight: 800; text-align: center; line-height: 1;
    margin-bottom: 0.5rem; font-family: var(--font-mono);
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    filter: drop-shadow(0 0 20px rgba(56, 189, 248, 0.4));
  }
  .result-confidence { text-align: center; font-size: 0.8rem; color: var(--text-muted); font-family: var(--font-mono); margin-bottom: 1.5rem; }
  .result-confidence span { color: var(--accent-green); font-weight: 700; font-size: 1.1rem; }
  .top-k-chart { display: flex; flex-direction: column; gap: 0.6rem; margin-top: 1rem; }
  .top-k-item { display: flex; align-items: center; gap: 0.75rem; }
  .top-k-char { width: 28px; text-align: center; font-family: var(--font-mono); font-weight: 700; font-size: 0.9rem; color: var(--text-primary); flex-shrink: 0; }
  .top-k-bar-wrap { flex: 1; height: 6px; background: rgba(48, 54, 61, 0.6); border-radius: 3px; overflow: hidden; }
  .top-k-bar { height: 100%; border-radius: 3px; transition: width 0.6s cubic-bezier(0.22, 1, 0.36, 1); background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)); }
  .top-k-pct { font-size: 0.7rem; font-family: var(--font-mono); color: var(--text-muted); width: 36px; text-align: right; flex-shrink: 0; }
  .char-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(28px, 1fr)); gap: 0.4rem; margin-top: 1rem; }
  .char-chip {
    aspect-ratio: 1; display: flex; align-items: center; justify-content: center;
    border-radius: 6px; font-size: 0.75rem; font-family: var(--font-mono); font-weight: 700;
    background: rgba(48, 54, 61, 0.4); color: var(--text-secondary); border: 1px solid transparent; transition: all 0.15s;
  }
  .training-grid { display: grid; grid-template-columns: 340px 1fr; gap: 1.5rem; }
  @media (max-width: 900px) { .training-grid { grid-template-columns: 1fr; } }
  .metrics-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
  @media (max-width: 600px) { .metrics-row { grid-template-columns: repeat(2, 1fr); } }
  .metric-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 1.1rem 1.25rem;
    backdrop-filter: blur(12px); transition: all 0.25s;
  }
  .metric-card:hover { border-color: rgba(56, 189, 248, 0.2); box-shadow: var(--glow-blue); }
  .metric-value { font-size: 1.5rem; font-weight: 800; font-family: var(--font-mono); color: var(--text-primary); line-height: 1.1; letter-spacing: -0.02em; }
  .metric-value.accent { color: var(--accent-blue); }
  .metric-value.success { color: var(--accent-green); }
  .metric-value.warning { color: var(--accent-amber); }
  .metric-value.neutral { color: var(--text-secondary); }
  .metric-label { font-size: 0.65rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); margin-top: 0.4rem; }
  .progress-bar-wrap { height: 6px; background: rgba(48, 54, 61, 0.6); border-radius: 3px; overflow: hidden; margin-top: 0.75rem; }
  .progress-bar {
    height: 100%; background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
    border-radius: 3px; transition: width 0.5s ease; position: relative; overflow: hidden;
  }
  .progress-bar::after {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
    animation: shimmer 1.8s ease infinite;
  }
  @keyframes shimmer { 0% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }
  .chart-wrap {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 1.25rem 1.5rem;
    backdrop-filter: blur(12px);
    height: 480px; min-height: 480px; max-height: 480px;
    display: flex; flex-direction: column; overflow: hidden;
  }
  .chart-canvas-container { position: relative; flex: 1; min-height: 0; overflow: hidden; }
  #training-chart { position: absolute; top: 0; left: 0; width: 100%; height: 340px; display: block; }
  .control-group { margin-bottom: 1.25rem; }
  .control-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: var(--text-muted); margin-bottom: 0.5rem; display: block; }
  .control-hint { font-size: 0.7rem; color: var(--text-muted); margin-top: 0.3rem; font-family: var(--font-mono); }
  input[type="range"] { width: 100%; height: 4px; -webkit-appearance: none; appearance: none; background: rgba(48, 54, 61, 0.8); border-radius: 2px; outline: none; cursor: pointer; }
  input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%; background: var(--accent-blue); box-shadow: 0 0 8px rgba(56, 189, 248, 0.5); cursor: pointer; transition: transform 0.15s; }
  input[type="range"]::-webkit-slider-thumb:hover { transform: scale(1.2); }
  .range-val { font-family: var(--font-mono); font-size: 0.8rem; color: var(--accent-blue); font-weight: 700; margin-left: 0.5rem; }
  .empty-state { text-align: center; padding: 3rem 2rem; color: var(--text-muted); }
  .empty-state-icon { font-size: 2.5rem; margin-bottom: 1rem; opacity: 0.5; }
  .empty-state-title { font-size: 0.9rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 0.5rem; }
  .empty-state-desc { font-size: 0.75rem; color: var(--text-muted); }
  #toast {
    position: fixed; bottom: 2rem; right: 2rem; padding: 0.85rem 1.25rem;
    border-radius: var(--radius-md); font-size: 0.8rem; font-weight: 600;
    backdrop-filter: blur(12px); z-index: 9999; opacity: 0;
    transform: translateY(16px); transition: all 0.3s ease; pointer-events: none; max-width: 320px;
  }
  #toast.show { opacity: 1; transform: translateY(0); pointer-events: auto; }
  #toast.success { background: rgba(74, 222, 128, 0.15); border: 1px solid rgba(74, 222, 128, 0.3); color: var(--accent-green); }
  #toast.error { background: rgba(248, 113, 113, 0.15); border: 1px solid rgba(248, 113, 113, 0.3); color: var(--accent-red); }
  #toast.info { background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.3); color: var(--accent-blue); }
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(48, 54, 61, 0.8); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
  .divider { border: none; height: 1px; background: linear-gradient(90deg, transparent, var(--border), transparent); margin: 1.25rem 0; }
  .fade-up { animation: fadeUp 0.5s ease both; }
  @keyframes fadeUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
  .stagger-1 { animation-delay: 0.05s; }
  .stagger-2 { animation-delay: 0.1s; }
  .stagger-3 { animation-delay: 0.15s; }
  .stagger-4 { animation-delay: 0.2s; }
</style>
</head>
<body>

<div id="neural-bg"><canvas id="bg-canvas"></canvas></div>

<div class="app">
  <header class="header">
    <div class="header-brand">
      <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
        <circle cx="18" cy="18" r="16" stroke="#38bdf8" stroke-width="1.5" fill="none" opacity="0.4"/>
        <circle cx="10" cy="12" r="3" fill="#38bdf8" opacity="0.8"/>
        <circle cx="26" cy="10" r="2.5" fill="#22d3ee" opacity="0.8"/>
        <circle cx="28" cy="22" r="2" fill="#4ade80" opacity="0.7"/>
        <circle cx="18" cy="26" r="2.5" fill="#38bdf8" opacity="0.6"/>
        <circle cx="8" cy="24" r="2" fill="#22d3ee" opacity="0.6"/>
        <line x1="10" y1="12" x2="26" y2="10" stroke="#38bdf8" stroke-width="0.8" opacity="0.3"/>
        <line x1="26" y1="10" x2="28" y2="22" stroke="#22d3ee" stroke-width="0.8" opacity="0.3"/>
        <line x1="28" y1="22" x2="18" y2="26" stroke="#4ade80" stroke-width="0.8" opacity="0.3"/>
        <line x1="18" y1="26" x2="8" y2="24" stroke="#38bdf8" stroke-width="0.8" opacity="0.3"/>
        <line x1="8" y1="24" x2="10" y2="12" stroke="#22d3ee" stroke-width="0.8" opacity="0.3"/>
        <line x1="10" y1="12" x2="18" y2="26" stroke="#38bdf8" stroke-width="0.6" opacity="0.2"/>
        <line x1="26" y1="10" x2="18" y2="26" stroke="#22d3ee" stroke-width="0.6" opacity="0.2"/>
      </svg>
      <span class="header-title">CNN 字符识别</span>
      <span class="header-badge">Neural Dark</span>
    </div>
    <div class="header-status">
      <div class="status-dot" id="status-dot"></div>
      <span id="status-text">连接中...</span>
    </div>
  </header>

  <main class="main">
    <div class="nav-tabs">
      <button class="nav-tab active" data-tab="inference">
        <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
        推理
      </button>
      <button class="nav-tab" data-tab="training">
        <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 20V10M6 20V4M18 20v-6"/></svg>
        训练
      </button>
    </div>

    <!-- ===== 推理面板 ===== -->
    <div class="panel active" id="panel-inference">
      <div class="inference-grid">
        <div class="card fade-up">
          <div class="card-title">
            <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/></svg>
            上传图片
          </div>
          <div class="control-label">选择模型架构</div>
          <select class="form-select" id="inf-model-select">
            <option value="detailed" selected>DetailedCNN (4.32M 参数)</option>
            <option value="simple">SimpleCNN (8.44M 参数)</option>
            <option value="resnet">ResNet (3.05M 参数)</option>
          </select>
          <div class="drop-zone" id="drop-zone">
            <div class="drop-zone-icon">
              <svg width="48" height="48" fill="none" stroke="#8b949e" stroke-width="1.5" viewBox="0 0 24 24"><path d="M4 14.899A7 7 0 1 1 15.1 6.5c-.737.73-1.315 1.52-1.672 2.35a3 3 0 0 1 2.087 1.35L16 11M8 17l-3.5-3.5M12 3v11M3 15h4"/></svg>
            </div>
            <div class="drop-zone-text">拖拽图片到这里，或 <strong>点击选择文件</strong></div>
            <div class="drop-zone-hint">支持 JPG, PNG, BMP - 64x64 灰度</div>
            <input type="file" id="file-input" accept="image/*" style="display:none">
          </div>
          <img id="preview-img" alt="preview">
          <button class="btn btn-primary btn-full" id="predict-btn" disabled style="margin-top:1rem">
            <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
            开始识别
          </button>
        </div>

        <div class="card fade-up stagger-1">
          <div class="card-title">
            <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M9 19v-6a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2zM17 9V5a2 2 0 0 0-2-2H9a2 2 0 0 0-2 2v4M21 15v2a2 2 0 0 1-2 2h-2a2 2 0 0 1-2-2v-2"/></svg>
            识别结果
          </div>
          <div id="result-empty" class="empty-state">
            <div class="empty-state-icon">
              <svg width="40" height="40" fill="none" stroke="#484f58" stroke-width="1.5" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
            </div>
            <div class="empty-state-title">等待上传图片</div>
            <div class="empty-state-desc">上传一张字符图片即可开始识别</div>
          </div>
          <div class="result-panel" id="result-content">
            <div class="result-char" id="result-char">-</div>
            <div class="result-confidence">置信度 <span id="result-conf">0%</span></div>
            <div class="top-k-chart" id="top-k-chart"></div>
          </div>
        </div>
      </div>

      <div class="card fade-up stagger-2" style="margin-top:1.5rem">
        <div class="card-title">
          <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M4 6h16M4 12h16M4 18h7"/></svg>
          支持的字符类别
        </div>
        <div class="char-grid" id="char-grid"></div>
      </div>
    </div>

    <!-- ===== 训练面板 ===== -->
    <div class="panel" id="panel-training">
      <div class="metrics-row">
        <div class="metric-card fade-up stagger-1">
          <div class="metric-value accent" id="m-status">空闲</div>
          <div class="metric-label">训练状态</div>
          <div class="progress-bar-wrap" id="progress-wrap" style="display:none">
            <div class="progress-bar" id="progress-bar" style="width:0%"></div>
          </div>
        </div>
        <div class="metric-card fade-up stagger-2">
          <div class="metric-value neutral" id="m-epoch">0 / 30</div>
          <div class="metric-label">当前轮次</div>
        </div>
        <div class="metric-card fade-up stagger-3">
          <div class="metric-value success" id="m-acc">---</div>
          <div class="metric-label">最佳准确率</div>
        </div>
        <div class="metric-card fade-up stagger-4">
          <div class="metric-value warning" id="m-lr">0.001</div>
          <div class="metric-label">学习率</div>
        </div>
      </div>

      <div class="training-grid">
        <div class="card fade-up">
          <div class="card-title">
            <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
            训练配置
          </div>
          <div class="control-group">
            <label class="control-label">模型架构</label>
            <select class="form-select" id="train-model-select">
              <option value="detailed">DetailedCNN (4.32M 参数)</option>
              <option value="simple">SimpleCNN (8.44M 参数)</option>
              <option value="resnet">ResNet (3.05M 参数)</option>
            </select>
          </div>
          <hr class="divider">
          <div class="control-group">
            <label class="control-label">训练轮次 <span class="range-val" id="epochs-val">30</span></label>
            <input type="range" id="epochs-slider" min="1" max="100" value="30">
            <div class="control-hint">当前: <span id="epochs-display">30</span> 轮</div>
          </div>
          <div class="control-group">
            <label class="control-label">批次大小</label>
            <select class="form-select" id="batch-select">
              <option value="32">32</option>
              <option value="64" selected>64</option>
              <option value="128">128</option>
              <option value="256">256</option>
            </select>
          </div>
          <div class="control-group">
            <label class="control-label">学习率 <span class="range-val" id="lr-val">0.001</span></label>
            <input type="range" id="lr-slider" min="1" max="5" step="1" value="3">
            <div class="control-hint" id="lr-display">0.001</div>
          </div>
          <hr class="divider">
          <div style="display:flex;gap:0.75rem;">
            <button class="btn btn-primary" id="start-btn" style="flex:1">
              <svg width="14" height="14" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
              开始训练
            </button>
            <button class="btn btn-danger" id="stop-btn" disabled>
              <svg width="14" height="14" fill="currentColor" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12"/></svg>
            </button>
          </div>
        </div>

        <div class="chart-wrap fade-up stagger-1">
          <div class="card-title" style="margin-bottom:1rem">
            <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M3 3v18h18"/><path d="M18 9l-5 5-4-4-3 3"/></svg>
            训练曲线
          </div>
          <div class="chart-canvas-container">
            <canvas id="training-chart"></canvas>
          </div>
          <div id="realtime-metrics" style="display:none;margin-top:1rem;">
            <hr class="divider">
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;text-align:center;">
              <div>
                <div style="font-size:1.1rem;font-weight:700;font-family:var(--font-mono);color:var(--accent-blue);" id="rt-train-loss">---</div>
                <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.06em;color:var(--text-muted);margin-top:0.2rem;">训练 Loss</div>
              </div>
              <div>
                <div style="font-size:1.1rem;font-weight:700;font-family:var(--font-mono);color:var(--text-secondary);" id="rt-val-loss">---</div>
                <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.06em;color:var(--text-muted);margin-top:0.2rem;">验证 Loss</div>
              </div>
              <div>
                <div style="font-size:1.1rem;font-weight:700;font-family:var(--font-mono);color:var(--accent-green);" id="rt-train-acc">---</div>
                <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.06em;color:var(--text-muted);margin-top:0.2rem;">训练 Acc</div>
              </div>
              <div>
                <div style="font-size:1.1rem;font-weight:700;font-family:var(--font-mono);color:var(--accent-amber);" id="rt-val-acc">---</div>
                <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.06em;color:var(--text-muted);margin-top:0.2rem;">验证 Acc</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </main>
</div>

<div id="toast"></div>

<script>
/* Neural background animation */
(function() {
  var canvas = document.getElementById('bg-canvas');
  var ctx = canvas.getContext('2d');
  var W, H, nodes = [], animFrame;

  function resize() {
    W = canvas.width = canvas.offsetWidth * devicePixelRatio;
    H = canvas.height = canvas.offsetHeight * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);
    initNodes();
  }

  function initNodes() {
    nodes = [];
    var count = Math.floor((window.innerWidth * window.innerHeight) / 18000);
    for (var i = 0; i < count; i++) {
      nodes.push({
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
        vx: (Math.random() - 0.5) * 0.25,
        vy: (Math.random() - 0.5) * 0.25,
        r: Math.random() * 2 + 1,
        a: Math.random() * 0.5 + 0.2
      });
    }
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i];
      n.x += n.vx; n.y += n.vy;
      if (n.x < 0) n.x = window.innerWidth;
      if (n.x > window.innerWidth) n.x = 0;
      if (n.y < 0) n.y = window.innerHeight;
      if (n.y > window.innerHeight) n.y = 0;
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(56,189,248,' + n.a + ')';
      ctx.fill();
    }
    for (var i = 0; i < nodes.length; i++) {
      for (var j = i + 1; j < nodes.length; j++) {
        var dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y;
        var dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.strokeStyle = 'rgba(56,189,248,' + (0.08 * (1 - dist / 120)) + ')';
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    animFrame = requestAnimationFrame(draw);
  }

  resize();
  draw();
  window.addEventListener('resize', resize);
})();

/* State */
var BACKEND = 'http://127.0.0.1:8000';
var currentModel = 'detailed';
var trainingInterval = null;
var chartInstance = null;
var uploadedFile = null;
var trainingHistory = [];

/* Tab switch */
document.querySelectorAll('.nav-tab').forEach(function(tab) {
  tab.addEventListener('click', function() {
    document.querySelectorAll('.nav-tab').forEach(function(t) { t.classList.remove('active'); });
    document.querySelectorAll('.panel').forEach(function(p) { p.classList.remove('active'); });
    tab.classList.add('active');
    document.getElementById('panel-' + tab.dataset.tab).classList.add('active');
  });
});

/* Model selector - 使用下拉菜单 */
document.getElementById('inf-model-select').addEventListener('change', function(e) { currentModel = e.target.value; });
document.getElementById('train-model-select').addEventListener('change', function(e) { currentModel = e.target.value; });

/* Char grid */
(function() {
  var grid = document.getElementById('char-grid');
  var chars = [];
  for (var i = 0; i < 10; i++) chars.push(String(i));
  for (var i = 0; i < 26; i++) chars.push(String.fromCharCode(97 + i));
  for (var i = 0; i < 26; i++) chars.push(String.fromCharCode(65 + i));
  chars.forEach(function(c) {
    var d = document.createElement('div');
    d.className = 'char-chip';
    d.textContent = c;
    grid.appendChild(d);
  });
})();

/* Drag and drop */
var dropZone = document.getElementById('drop-zone');
var fileInput = document.getElementById('file-input');
var previewImg = document.getElementById('preview-img');
var predictBtn = document.getElementById('predict-btn');

dropZone.addEventListener('click', function() { fileInput.click(); });
dropZone.addEventListener('dragover', function(e) { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', function() { dropZone.classList.remove('drag-over'); });
dropZone.addEventListener('drop', function(e) {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  var f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});

fileInput.addEventListener('change', function(e) {
  var f = e.target.files[0];
  if (f) handleFile(f);
});

function handleFile(file) {
  uploadedFile = file;
  var reader = new FileReader();
  reader.onload = function(e) {
    previewImg.src = e.target.result;
    previewImg.style.display = 'block';
    predictBtn.disabled = false;
    document.getElementById('result-empty').style.display = 'block';
    document.getElementById('result-content').classList.remove('show');
  };
  reader.readAsDataURL(file);
}

/* Inference */
predictBtn.addEventListener('click', function() {
  if (!uploadedFile) return;
  predictBtn.disabled = true;
  predictBtn.innerHTML = '<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg> 识别中...';

  var form = new FormData();
  form.append('file', uploadedFile);

  fetch(BACKEND + '/predict/?net_type=' + encodeURIComponent(currentModel), { method: 'POST', body: form })
    .then(function(res) { return res.json(); })
    .then(function(data) {
      showResult(data.prediction);
      toast('识别成功', 'success');
    })
    .catch(function(e) {
      toast('推理失败: ' + e.message, 'error');
    })
    .finally(function() {
      predictBtn.disabled = false;
      predictBtn.innerHTML = '<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M13 10V3L4 14h7v7l9-11h-7z"/></svg> 开始识别';
    });
});

function showResult(pred) {
  document.getElementById('result-empty').style.display = 'none';
  var content = document.getElementById('result-content');
  content.classList.add('show');
  document.getElementById('result-char').textContent = pred.class;
  document.getElementById('result-conf').textContent = (pred.confidence * 100).toFixed(1) + '%';

  var chart = document.getElementById('top-k-chart');
  chart.innerHTML = '';
  pred.top_k.forEach(function(item, i) {
    var div = document.createElement('div');
    div.className = 'top-k-item';
    div.innerHTML = '<div class="top-k-char">' + item.class + '</div>' +
      '<div class="top-k-bar-wrap"><div class="top-k-bar" style="width:0%"></div></div>' +
      '<div class="top-k-pct">' + (item.confidence * 100).toFixed(1) + '%</div>';
    chart.appendChild(div);
    requestAnimationFrame(function() {
      div.querySelector('.top-k-bar').style.width = (item.confidence * 100) + '%';
    });
  });
}

/* Training controls */
var epochsSlider = document.getElementById('epochs-slider');
var epochsVal = document.getElementById('epochs-val');
var epochsDisplay = document.getElementById('epochs-display');

epochsSlider.addEventListener('input', function() {
  epochsVal.textContent = epochsSlider.value;
  epochsDisplay.textContent = epochsSlider.value;
});

var lrSlider = document.getElementById('lr-slider');
var lrVal = document.getElementById('lr-val');
var lrDisplay = document.getElementById('lr-display');
var LR_MAP = { 1: 0.0001, 2: 0.0005, 3: 0.001, 4: 0.005, 5: 0.01 };

lrSlider.addEventListener('input', function() {
  var v = LR_MAP[parseInt(lrSlider.value)];
  lrVal.textContent = v;
  lrDisplay.textContent = v + ' (默认)';
});

/* Training start/stop */
var startBtn = document.getElementById('start-btn');
var stopBtn = document.getElementById('stop-btn');

startBtn.addEventListener('click', startTraining);
stopBtn.addEventListener('click', stopTraining);

function getBatchSize() {
  return parseInt(document.getElementById('batch-select').value);
}

function startTraining() {
  fetch(BACKEND + '/train/start/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      net_type: currentModel,
      epochs: parseInt(epochsSlider.value),
      batch_size: getBatchSize(),
      learning_rate: LR_MAP[parseInt(lrSlider.value)]
    })
  })
  .then(function(res) { return res.json(); })
  .then(function(data) {
    if (data.error) throw new Error(data.detail || data.error);
    toast('训练已启动', 'info');
    startBtn.disabled = true;
    stopBtn.disabled = false;
    trainingHistory = [];
    updateTrainingUI('running');
    pollTraining();
  })
  .catch(function(e) {
    toast('启动失败: ' + e.message, 'error');
  });
}

function stopTraining() {
  fetch(BACKEND + '/train/stop/', { method: 'POST' })
    .then(function() { toast('训练已停止', 'info'); })
    .catch(function(e) { toast('停止失败: ' + e.message, 'error'); });
}

function pollTraining() {
  if (startBtn.disabled === false) return;
  Promise.all([
    fetch(BACKEND + '/train/status/'),
    fetch(BACKEND + '/train/metrics/')
  ])
  .then(function(results) {
    return Promise.all(results.map(function(r) { return r.json(); }));
  })
  .then(function(data) {
    var status = data[0];
    var metrics = data[1];
    updateTrainingUI(status.status, status, metrics);
    if (metrics.history && metrics.history.length > trainingHistory.length) {
      trainingHistory = metrics.history;
      updateChart(trainingHistory);
    }
    if (status.status === 'running' || status.status === 'idle') {
      trainingInterval = setTimeout(pollTraining, 2000);
    } else {
      startBtn.disabled = false;
      stopBtn.disabled = true;
      if (status.status === 'completed') toast('训练完成!', 'success');
      else if (status.status === 'error') toast('训练出错', 'error');
    }
  })
  .catch(function() {});
}

function updateTrainingUI(status, statusData, metrics) {
  if (!status) return;
  var labels = { idle: '空闲', running: '训练中', completed: '已完成', stopped: '已停止', error: '出错' };
  document.getElementById('m-status').textContent = labels[status] || status;

  var dot = document.getElementById('status-dot');
  if (status === 'running' && statusData) {
    dot.className = 'status-dot training';
    document.getElementById('progress-wrap').style.display = 'block';
    var p = (statusData.epoch / (statusData.epochs || 1)) * 100;
    document.getElementById('progress-bar').style.width = p + '%';
    document.getElementById('realtime-metrics').style.display = 'block';
  } else {
    dot.className = (status === 'completed') ? 'status-dot connected' : 'status-dot';
    document.getElementById('progress-wrap').style.display = 'none';
    document.getElementById('realtime-metrics').style.display = 'none';
  }

  if (metrics && typeof metrics === 'object') {
    var epoch = metrics.epoch !== undefined ? metrics.epoch : 0;
    var epochs = metrics.epochs !== undefined ? metrics.epochs : 1;
    document.getElementById('m-epoch').textContent = epoch + ' / ' + epochs;
    document.getElementById('m-acc').textContent = metrics.best_acc > 0 ? metrics.best_acc.toFixed(2) + '%' : '---';
    document.getElementById('m-lr').textContent = metrics.lr ? metrics.lr.toFixed(4) : '---';
    if (metrics.train_loss !== undefined) document.getElementById('rt-train-loss').textContent = metrics.train_loss.toFixed(4);
    if (metrics.val_loss !== undefined) document.getElementById('rt-val-loss').textContent = metrics.val_loss.toFixed(4);
    if (metrics.train_acc !== undefined) document.getElementById('rt-train-acc').textContent = metrics.train_acc.toFixed(1) + '%';
    if (metrics.val_acc !== undefined) document.getElementById('rt-val-acc').textContent = metrics.val_acc.toFixed(1) + '%';
  }
}

/* Chart */
function getChartWidth() {
  var canvas = document.getElementById('training-chart');
  var container = canvas.parentElement;
  var w = container.clientWidth;
  if (!w || w < 100 || w > 5000) w = 800;
  return w;
}

function updateChart(history) {
  var canvas = document.getElementById('training-chart');
  if (!history || history.length === 0) return;

  var epochs = history.map(function(h) { return h.epoch; });
  var trainLoss = history.map(function(h) { return h.train_loss; });
  var valLoss = history.map(function(h) { return h.val_loss; });
  var trainAcc = history.map(function(h) { return h.train_acc; });
  var valAcc = history.map(function(h) { return h.val_acc; });

  var chartW = getChartWidth();
  canvas.style.height = '340px';
  canvas.style.width = chartW + 'px';
  canvas.height = 340;
  canvas.width = chartW;

  if (chartInstance) {
    chartInstance.data.labels = epochs;
    chartInstance.data.datasets[0].data = trainLoss;
    chartInstance.data.datasets[1].data = valLoss;
    chartInstance.data.datasets[2].data = trainAcc;
    chartInstance.data.datasets[3].data = valAcc;
    chartInstance.resize(chartW, 340);
    chartInstance.update('none');
    return;
  }

  chartInstance = new Chart(canvas, {
    type: 'line',
    data: {
      labels: epochs,
      datasets: [
        { label: '训练损失', data: trainLoss, borderColor: '#38bdf8', backgroundColor: 'rgba(56,189,248,0.05)', borderWidth: 2, pointRadius: 3, pointBackgroundColor: '#38bdf8', tension: 0.4, fill: true },
        { label: '验证损失', data: valLoss, borderColor: '#8b949e', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 3, pointBackgroundColor: '#8b949e', tension: 0.4, borderDash: [4, 4] },
        { label: '训练准确率', data: trainAcc, borderColor: '#4ade80', backgroundColor: 'rgba(74,222,128,0.05)', borderWidth: 2, pointRadius: 3, pointBackgroundColor: '#4ade80', tension: 0.4, fill: true },
        { label: '验证准确率', data: valAcc, borderColor: '#fbbf24', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 3, pointBackgroundColor: '#fbbf24', tension: 0.4, borderDash: [4, 4] }
      ]
    },
    options: {
      responsive: false,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          position: 'top',
          labels: { color: '#8b949e', font: { family: 'Inter, system-ui', size: 11 }, usePointStyle: true, pointStyleWidth: 12, padding: 16 }
        },
        tooltip: {
          backgroundColor: 'rgba(13,17,23,0.95)',
          borderColor: 'rgba(48,54,61,0.8)',
          borderWidth: 1,
          titleColor: '#e6edf3',
          bodyColor: '#8b949e',
          padding: 10
        }
      },
      scales: {
        x: { title: { display: true, text: '训练轮次', color: '#484f58', font: { family: 'Inter', size: 11 } }, grid: { color: 'rgba(48,54,61,0.4)' }, ticks: { color: '#484f58', size: 10 } },
        y: { title: { display: false }, grid: { color: 'rgba(48,54,61,0.4)' }, ticks: { color: '#484f58', size: 10 } }
      }
    }
  });
}

/* Toast */
var toastTimeout;
function toast(msg, type) {
  type = type || 'info';
  var el = document.getElementById('toast');
  el.textContent = msg;
  el.className = type + ' show';
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(function() { el.classList.remove('show'); }, 3000);
}

/* Backend check */
function checkBackend() {
  var dot = document.getElementById('status-dot');
  var text = document.getElementById('status-text');
  fetch(BACKEND + '/train/status/', { method: 'GET', signal: AbortSignal.timeout(3000) })
    .then(function() {
      dot.className = 'status-dot connected';
      text.textContent = '已连接';
    })
    .catch(function() {
      dot.className = 'status-dot';
      text.textContent = '后端离线';
    });
}
checkBackend();
setInterval(checkBackend, 15000);

/* Init empty chart - 预创建 4 个空 datasets，后续 updateChart 直接复用 */
(function initEmptyChart() {
  var canvas = document.getElementById('training-chart');
  var chartW = getChartWidth();
  canvas.style.height = '340px';
  canvas.style.width = chartW + 'px';
  canvas.height = 340;
  canvas.width = chartW;
  chartInstance = new Chart(canvas, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: '训练损失', data: [], borderColor: '#38bdf8', backgroundColor: 'rgba(56,189,248,0.05)', borderWidth: 2, pointRadius: 3, pointBackgroundColor: '#38bdf8', tension: 0.4, fill: true },
        { label: '验证损失', data: [], borderColor: '#8b949e', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 3, pointBackgroundColor: '#8b949e', tension: 0.4, borderDash: [4, 4] },
        { label: '训练准确率', data: [], borderColor: '#4ade80', backgroundColor: 'rgba(74,222,128,0.05)', borderWidth: 2, pointRadius: 3, pointBackgroundColor: '#4ade80', tension: 0.4, fill: true },
        { label: '验证准确率', data: [], borderColor: '#fbbf24', backgroundColor: 'transparent', borderWidth: 2, pointRadius: 3, pointBackgroundColor: '#fbbf24', tension: 0.4, borderDash: [4, 4] }
      ]
    },
    options: {
      responsive: false,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      plugins: {
        legend: {
          position: 'top',
          labels: { color: '#8b949e', font: { family: 'Inter, system-ui', size: 11 }, usePointStyle: true, pointStyleWidth: 12, padding: 16 }
        },
        tooltip: {
          backgroundColor: 'rgba(13,17,23,0.95)',
          borderColor: 'rgba(48,54,61,0.8)',
          borderWidth: 1,
          titleColor: '#e6edf3',
          bodyColor: '#8b949e',
          padding: 10
        }
      },
      scales: {
        x: { title: { display: true, text: '训练轮次', color: '#484f58', font: { family: 'Inter', size: 11 } }, grid: { color: 'rgba(48,54,61,0.4)' }, ticks: { color: '#484f58', size: 10 } },
        y: { title: { display: false }, grid: { color: 'rgba(48,54,61,0.4)' }, ticks: { color: '#484f58', size: 10 } }
      }
    }
  });
})();
</script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="CNN 字符识别前端服务器")
    parser.add_argument('--port', type=int, default=8501)
    parser.add_argument('--backend', type=str, default='http://127.0.0.1:8000')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(base_dir, '.tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    html_path = os.path.join(tmp_dir, 'frontend.html')

    content = HTML_CONTENT.replace("var BACKEND = 'http://127.0.0.1:8000';", "var BACKEND = '" + args.backend + "';")

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("启动前端服务: http://127.0.0.1:" + str(args.port))
    print("请确保后端运行在:", args.backend)
    print()
    print("访问 http://127.0.0.1:" + str(args.port) + " 打开前端")

    os.chdir(base_dir)
    server = HTTPServer(('127.0.0.1', args.port), SPAHandler)
    server.serve_forever()


if __name__ == '__main__':
    main()
