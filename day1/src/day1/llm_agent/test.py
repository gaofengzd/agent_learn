# import torch
#
# def get_device():
#     """检测是否有可用的CUDA GPU，有则返回'cuda'，否则'cpu'"""
#     if torch.cuda.is_available():
#         print(f"✅ 检测到 GPU: {torch.cuda.get_device_name(0)}，将使用 GPU 加速")
#         return 'cuda'
#     else:
#         print("ℹ️ 未检测到 CUDA GPU，将使用 CPU 运行（速度会慢一些）")
#         return 'cpu'
#
# # 获取设备
# device = get_device()


from modelscope import snapshot_download

# 这会在命令行打印非常清晰的进度条
model_dir = snapshot_download('AI-ModelScope/bge-reranker-v2-m3', cache_dir =r"E:\00project\02agent\models")
print(f"模型下载到了本地路径：{model_dir}")