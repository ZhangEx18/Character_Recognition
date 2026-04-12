# frontend.py
import streamlit as st
import requests

# FastAPI 后端的地址
BACKEND_URL = "http://127.0.0.1:8000/predict/"

st.title("我的神经网络本地测试平台")

# 创建文件上传组件
uploaded_file = st.file_uploader("请上传一张测试图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
	# 在界面上展示上传的图片
	st.image(uploaded_file, caption="待测试图片", use_column_width=True)

	# 增加一个按钮来触发预测
	if st.button("开始预测"):
		with st.spinner("模型推理中，请稍候..."):
			# 准备要发送给后端的数据
			files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

			try:
				# 向 FastAPI 发送 POST 请求
				response = requests.post(BACKEND_URL, files=files)
				response.raise_for_status()  # 检查请求是否成功

				# 解析返回的结果并展示
				result = response.json()
				st.success("预测成功！")
				st.json(result["prediction"])

			except requests.exceptions.RequestException as e:
				st.error(f"请求后端失败，请检查后端服务是否已启动。错误信息: {e}")