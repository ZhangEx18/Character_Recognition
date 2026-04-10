"""
实时摄像头预测脚本
运行方式: python -m tools.camera_app

功能：
1. 吊起 Mac 前置摄像头
2. 实时捕获 ROI (感兴趣区域) 并在屏幕绘制参考框
3. 动态色彩反转与二值化处理 (适配现实世界的白纸黑字)
4. 异步降频推理 (保证视频流丝滑不卡顿)
"""

import cv2
import argparse
from PIL import Image

# 【核心修改 1】：从 src 导入大脑
from src import Predictor


def run_camera_app(model_path: str):
    print("=" * 60)
    print("🚀 正在加载神经网络大脑...")
    print("=" * 60)

    # 初始化预测器（内部会自动调度 MPS 苹果硅加速）
    try:
       predictor = Predictor(model_path)
    except FileNotFoundError:
       print(f"找不到权重文件：{model_path}")
       return

    # 调用 Mac 默认摄像头 (索引通常为 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
       print("无法打开摄像头！")
       print("解决办法: 请前往 Mac 的 [系统设置] -> [隐私与安全性] -> [摄像头] 中授权终端/IDE访问。")
       return

    print("\n✅ 摄像头已开启！")
    print("👉 请将写有字符的纸张对准屏幕中央的红框。")
    print("🛑 按下键盘上的 'q' 键退出程序。")

    # 状态缓存变量（用于降频推理）
    frame_counter = 0
    process_every_n_frames = 5  # 每 5 帧推理一次，剩下的时间显示缓存结果，防止画面掉帧
    current_text = "Wait..."
    current_color = (0, 255, 255)  # 黄色

    while True:
       ret, frame = cap.read()
       if not ret:
          print("摄像头画面流中断。")
          break

       # Mac 的摄像头很多时候默认是镜像的，需要水平翻转一下，否则拿在手里的字是反的
       ret, frame = cap.read()

       # ✅ 关键：旋转成“竖屏逻辑”
       frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

       # ❌ 先不要镜像
       # frame = cv2.flip(frame, 1)

       # 1. 计算中心准星框的坐标
       height, width, _ = frame.shape
       box_size = 250  # 框的大小
       x1 = (width - box_size) // 2
       y1 = (height - box_size) // 2
       x2 = x1 + box_size
       y2 = y1 + box_size

       # 画出红色的 ROI 区域框
       cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

       # 2. 截取框内的图像
       roi = frame[y1:y2, x1:x2]

       frame_counter += 1

       # 3. 降频推理逻辑：只在特定的帧数进行神经网络计算
       if frame_counter % process_every_n_frames == 0:
          # 【图像处理核心步骤】
          # A. 转为灰度图
          roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

          # B. 二值化与反色 (极其关键)
          # 现实中是"白纸黑字"，但神经网络通常喜欢"黑底白字"。
          # THRESH_BINARY_INV 会将低于阈值（黑色笔迹）的变成白色，高于阈值（白纸）的变成黑色。
          # 这里使用 Otsu 自动阈值算法，适应不同的室内光线。
          _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

          # C. 将 OpenCV 图像格式 (Numpy) 转为 PyTorch 友好的 PIL 格式
          pil_img = Image.fromarray(roi_thresh).convert('L')

          try:
             # 调用 predictor 初始化好的 transform 管道进行缩放和归一化
             tensor_img = predictor.transform(pil_img)

             # 增加 batch 维度: (1, 64, 64) -> (1, 1, 64, 64)
             tensor_img = tensor_img.unsqueeze(0)

             # 执行预测
             result = predictor.predict(tensor_img)

             char = result['class_name']
             conf = result['confidence']

             # 根据置信度改变字体颜色 (绿>80% > 黄>50% > 红)
             if conf > 0.8:
                current_color = (0, 255, 0)  # 绿
             elif conf > 0.5:
                current_color = (0, 255, 255)  # 黄
             else:
                current_color = (0, 0, 255)  # 红

             current_text = f"Char: {char} ({conf:.1%})"

          except Exception as e:
             print(f"推理异常: {e}")

       # 4. 将预测结果渲染在屏幕上（不管当前帧有没有进行推理，都把结果画上去）
       # 文字带个黑色描边，防止在白纸背景下看不清
       cv2.putText(frame, current_text, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
       cv2.putText(frame, current_text, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, current_color, 2)

       # 5. 实时显示画面
       cv2.imshow('Mac Camera - AI Character Recognition', frame)

       # 6. 监听键盘事件，按 'q' 退出
       if cv2.waitKey(1) & 0xFF == ord('q'):
          break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    # 针对 macOS 的小修复：有时窗口关不掉，多调几次 waitKey 刷新事件队列
    for i in range(5):
       cv2.waitKey(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='启动摄像头进行实时字符识别')

    # 【核心修改 2】：默认路径改为相对于根目录
    parser.add_argument(
       '--model', '-m',
       type=str,
       default='checkpoints/best_model.pth',
       help='模型权重文件的路径'
    )

    args = parser.parse_args()
    run_camera_app(args.model)