from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

# 指定你希望下载权重到的目录
download_path = r"D:\Program Files (x86)\weight" # 注意 r"" 表示原始字符串，避免反斜杠的转义问题

# 确保目录存在。如果不存在，会自动创建。
os.makedirs(download_path, exist_ok=True)

# 选择你想要的 GPT-2 模型版本
model_name = "gpt2" # 或者 "gpt2-large", "gpt2-medium", "gpt2-xl"

print(f"尝试将 GPT-2 '{model_name}' 模型权重下载到: {download_path}")

# 加载 tokenizer（分词器），并指定下载路径
tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=download_path)

# 加载模型（这将自动下载权重到指定路径）
model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=download_path)

print(f"\nGPT-2 模型 '{model_name}' 及其权重已成功下载并加载。")
print(f"权重文件现在应该在以下目录中找到: {download_path}")