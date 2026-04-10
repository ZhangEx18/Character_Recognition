"""
实时摄像头预测脚本 (iPhone 连续互通摄像头优化版)
运行方式: python -m tools.camera_app --net resnet

功能：
1. 实时捕获中心红框区域 (ROI)
2. 针对 iPhone 连续互通摄像头进行了【镜像与姿势矫正】
3. 动态色彩反转与二值化处理 (适配白纸黑字)
4. 异步降频推理，保证视频流在 Mac 上运行丝滑
"""

import cv2
import argparse
from PIL import Image

# 从项目核心包导入预测引擎
from src import Predictor


def run_camera_app(model_path: str, net_type: str):
    print("=" * 60)
    print(f"🚀 正在加载 {net_type.upper()} 预测引擎...")
    print("=" * 60)

    # 1. 初始化预测器 (根据参数加载对应的模型架构)
    try:
        predictor = Predictor(model_path, net_type=net_type)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 2. 开启摄像头 (iPhone 通常映射在索引 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头！")
        return

    print("\n✅ 摄像头已开启！")
    print("👉 调整提示：确保屏幕上的按键字母是【正着】的，不是反向的。")
    print("🛑 按下键盘上的 'q' 键退出程序。")

    # 状态缓存与降频配置
    frame_counter = 0
    process_every_n_frames = 5  # 每 5 帧进行一次 AI 推理
    current_text = "Scanning..."
    current_color = (0, 255, 255)

    while True:
        # 修正：只读取一次，解决掉帧问题
        ret, frame = cap.read()
        if not ret:
            break

        # ============================================================
        # 【镜像与姿势矫正区】
        # 根据您的反馈：“上下对，左右反”，默认开启水平翻转
        # ============================================================

        # 修复左右反向：执行水平镜像翻转 (绕 Y 轴)
        frame = cv2.flip(frame, 1)

        # 如果您的 iPhone 是竖屏放置导致画面偏转，请取消下面对应行的注释：
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # ============================================================

        # 1. 计算中心准星框坐标
        h, w, _ = frame.shape
        box_size = 250
        x1, y1 = (w - box_size) // 2, (h - box_size) // 2
        x2, y2 = x1 + box_size, y1 + box_size

        # 绘制红色 ROI 引导框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 2. 推理逻辑
        frame_counter += 1
        if frame_counter % process_every_n_frames == 0:
            # 截取框内图像
            roi = frame[y1:y2, x1:x2]

            # A. 图像预处理：灰度化 -> 二值化反色 (将白纸黑字转为黑底白字)
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # B. 格式转换：Numpy 转 PIL 再转 Tensor
            pil_img = Image.fromarray(roi_thresh).convert('L')
            tensor_img = predictor.transform(pil_img)

            try:
                # C. 执行预测
                res = predictor.predict(tensor_img)
                char = res['class_name']
                conf = res['confidence']

                # D. 动态设置颜色 (置信度越高越绿)
                if conf > 0.8:
                    current_color = (0, 255, 0)  # 绿色
                elif conf > 0.5:
                    current_color = (0, 255, 255)  # 黄色
                else:
                    current_color = (0, 0, 255)  # 红色

                current_text = f"Char: {char} ({conf:.1%})"
            except Exception as e:
                print(f"推理引擎异常: {e}")

        # 3. 渲染预测结果
        cv2.putText(frame, current_text, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4)
        cv2.putText(frame, current_text, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, current_color, 2)

        # 4. 显示画面
        cv2.imshow('CNN Character Recognition', frame)

        # 监听键盘
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 资源释放
    cap.release()
    cv2.destroyAllWindows()
    for i in range(5):
        cv2.waitKey(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='启动实时摄像头字符识别')

    # 参数：权重路径
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='checkpoints/best_model.pth',
        help='模型权重物理路径'
    )

    # 参数：模型架构类型 (resnet / detailed / simple)
    parser.add_argument(
        '--net',
        type=str,
        default='detailed',
        choices=['simple', 'detailed', 'resnet'],
        help='必须与训练该权重时的模型类型一致'
    )

    args = parser.parse_args()
    run_camera_app(args.model, args.net)