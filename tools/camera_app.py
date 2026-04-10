"""
实时摄像头预测脚本 (iPhone 连续互通摄像头优化版)

【系统架构定位】
本模块是整个 CNN 系统的“眼睛”与“实战前线”。它负责吊起物理硬件（摄像头），
捕获真实世界的光学信号，将其清洗、过滤、预处理后，源源不断地喂给后端推理引擎，
并将 AI 的思考结果实时渲染在屏幕上。

【核心工程特性】
1. 性能隔离：采用“异步降频推理”策略，解耦摄像头帧率 (30fps) 与 AI 推理帧率，杜绝画面卡顿。
2. 光学清洗：集成 Otsu 自适应二值化算法，无视室内光线明暗，自动剥离背景。
3. 空间矫正：针对苹果生态 (Mac + iPhone 连续互通) 独特的底层流机制，彻底修复镜像与重力偏转问题。
"""

import cv2
import argparse
from PIL import Image

# 从项目核心包导入我们封装好的面向对象预测引擎
from src import Predictor


def run_camera_app(model_path: str, net_type: str):
    print("=" * 60)
    print(f"🚀 正在为 {net_type.upper()} 引擎分配物理摄像头...")
    print("=" * 60)

    # ============================================================
    # 1. 挂载 AI 预测引擎
    # ============================================================
    try:
        # 内部会自动识别并调用 Mac 的 MPS 芯片进行硬件加速
        predictor = Predictor(model_path, net_type=net_type)
    except Exception as e:
        print(f"【致命错误】加载模型失败，请检查权重文件是否存在: {e}")
        return

    # ============================================================
    # 2. 硬件层：初始化视频流
    # ============================================================
    # 索引 0 通常代表系统默认的首选摄像头。
    # 当 iPhone 靠近并开启“连续互通”时，Mac 会自动将 iPhone 镜头顶替到索引 0 的位置。
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("【拦截】无法打开摄像头！")
        print("💡 解决办法: 请前往 Mac 的 [系统设置] -> [隐私与安全性] -> [摄像头] 中授权终端/IDE访问。")
        return

    print("\n✅ 摄像头硬件链路已打通！")
    print("👉 调整提示：请确保屏幕上的按键或字母是【正着】的，不是像照镜子一样反着的。")
    print("🛑 按下键盘上的小写 'q' 键退出程序。")

    # ============================================================
    # 3. 状态寄存器初始化
    # ============================================================
    frame_counter = 0               # 帧数计数器 (用于降频计算)
    process_every_n_frames = 5      # 降频阈值：每 5 帧才让 AI 动一次脑子 (相当于 6 FPS 的推理速度)
    current_text = "Scanning..."  # 屏幕上显示的默认文字
    current_color = (0, 255, 255)   # 默认 UI 颜色 (黄色)

    # ============================================================
    # 4. 主事件循环 (Main Event Loop)
    # ============================================================
    while True:
        # A. 抽取单帧画面
        # ret 是布尔值，代表是否成功抓取；frame 是三维 Numpy 数组 (H, W, Channels)
        ret, frame = cap.read()
        if not ret:
            print("摄像头画面流中断，跳出循环。")
            break

        # B. 【光学空间矫正】 (iPhone 适配专区)
        # 物理世界的摄像头默认是“镜像”的（为了让人看着像照镜子一样自然）。
        # 但 AI 是个死脑筋，它不认识镜像反转的字符（比如把 p 看成 q）。
        # cv2.flip(frame, 1) 的 1 代表绕 Y 轴进行水平翻转，强行把镜像拉回现实视角。
        frame = cv2.flip(frame, 1)

        # 备用方案：如果您的 iPhone 是竖屏架在屏幕上导致画面转了 90 度，请取消下面对应的注释：
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # C. 计算感兴趣区域 (ROI - Region of Interest)
        # 我们不需要识别整个屏幕，只需要识别正中间那一小块。这样可以排除背景干扰。
        h, w, _ = frame.shape
        box_size = 250  # 正方形识别框的边长 (像素)
        x1, y1 = (w - box_size) // 2, (h - box_size) // 2  # 左上角坐标
        x2, y2 = x1 + box_size, y1 + box_size              # 右下角坐标

        # 在全局画面上画出一个红色的粗框，引导用户把字放在这里
        # 参数: (图像, 左上角, 右下角, BGR颜色(红), 线条粗细)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # D. 异步降频推理 (防卡顿核心逻辑)
        # 为什么不每一帧都推理？因为摄像头每秒 30 帧，如果在主线程里每秒跑 30 次神经网络，
        # 画面会因为来不及计算而产生严重的拖影和卡顿。
        frame_counter += 1
        if frame_counter % process_every_n_frames == 0:

            # [1] 切片提取：只把红框里的像素切下来送给 AI
            roi = frame[y1:y2, x1:x2]

            # [2] 颜色空间转换：剥离彩色光影，转为纯灰度图 (因为我们的模型输入通道设为了 1)
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # [3] 图像二值化与反色处理 (极度重要！！！)
            # 现实中我们是“白纸写黑字”，但在 MNIST 风格的数据集中，底色是黑的，字是白的。
            # THRESH_BINARY_INV：将低于阈值（黑字）变白，高于阈值（白纸）变黑。
            # THRESH_OTSU：大津算法，自动计算当前环境光线下的最佳切割阈值，无视阴影。
            _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # [4] 数据结构桥接：OpenCV(Numpy array) -> Python(PIL Image) -> Tensor
            pil_img = Image.fromarray(roi_thresh).convert('L')
            tensor_img = predictor.transform(pil_img)

            try:
                # [5] 核心预测
                res = predictor.predict(tensor_img)
                char = res['class_name']
                conf = res['confidence']

                # [6] 动态 UI 视觉反馈 (置信度红绿灯)
                if conf > 0.8:
                    current_color = (0, 255, 0)    # 绿色：极度自信
                elif conf > 0.5:
                    current_color = (0, 255, 255)  # 黄色：勉强认出，可能存在歧义
                else:
                    current_color = (0, 0, 255)    # 红色：完全瞎猜

                current_text = f"Char: {char} ({conf:.1%})"
            except Exception as e:
                print(f"推理引擎执行异常: {e}")

        # E. UI 渲染层
        # OpenCV 写字默认没有背景色。如果背景正好是白的，白字就看不见了。
        # 技巧：先画一个粗的黑色字体当“描边”，再在上面画一个细的彩色字体当“字芯”。
        cv2.putText(frame, current_text, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4)
        cv2.putText(frame, current_text, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, current_color, 2)

        # F. 屏幕推流
        cv2.imshow('CNN Character Recognition [Live]', frame)

        # G. 监听中断信号
        # waitKey(1) 表示等待 1 毫秒获取键盘输入。0xFF 掩码是为了跨平台兼容 ASCII 码。
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ============================================================
    # 5. 内存与硬件资源回收
    # ============================================================
    cap.release()               # 归还摄像头控制权给系统
    cv2.destroyAllWindows()     # 砸碎所有显示窗口

    # 【Mac 专属避坑】：Mac 的窗口管理器有时反应慢，加一个废循环清空事件队列，确保窗口真关掉了。
    for i in range(5):
        cv2.waitKey(1)


# ============================================================
# 命令行终端 (CLI) 调度入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='启动实时摄像头字符识别系统')

    # 参数绑定：模型权重物理路径
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='checkpoints/best_model.pth',
        help='编译好的模型权重 (.pth) 物理路径'
    )

    # 参数绑定：架构锁芯
    # 【避坑】：如果您拿着 ResNet 的权重去开 DetailedCNN 的门，会直接报字典不匹配错误。
    parser.add_argument(
        '--net',
        type=str,
        default='detailed',
        choices=['simple', 'detailed', 'resnet'],
        help='声明该权重文件是由哪种架构训练出来的'
    )

    args = parser.parse_args()

    # 扣动扳机，启动主程序
    run_camera_app(args.model, args.net)