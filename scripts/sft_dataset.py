import json
import os
import urllib.request
import torch
from torch.utils.data import Dataset
import tiktoken



def download_and_load_file(file_path, url):
	"""
	下载并加载指定URL的文件。如果文件不存在，则从URL下载；否则，直接从本地文件加载。
	文件内容被解码为UTF-8并解析为JSON。

	Args:
		file_path (str): 本地文件的路径。
		url (str): 文件的URL。

	Returns:
		dict: 解析后的JSON数据。
	"""
	if not os.path.exists(file_path):
		with urllib.request.urlopen(url) as response:
			text_data = response.read().decode('utf-8')
		with open(file_path, 'w', encoding = "utf-8") as file:
			file.write(text_data)
	else:
		with open(file_path, 'r', encoding = "utf-8") as file:
			text_data = file.read()
	data = json.loads(text_data)
	return data

def format_input(entry):
	"""
	根据给定的条目（entry）格式化输入文本。
	它将指令和可选的输入内容组合成一个统一的字符串。

	Args:
		entry (dict): 包含 'instruction' 和 'input' 键的字典。

	Returns:
		str: 格式化后的输入文本。
	"""
	instruction_text = (
		f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"		
	)
	input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
	return instruction_text + input_text

class InstructionDataset(Dataset):
	"""
	用于处理指令数据集的PyTorch Dataset类。
	它将指令、输入和响应组合成完整的文本，并使用tokenizer进行编码。
	"""
	def __init__(self, data, tokenizer):
		"""
		初始化InstructionDataset。

		Args:
			data (list): 包含指令、输入和输出的字典列表。
			tokenizer: 用于编码文本的tokenizer。
		"""
		self.data = data
		self.encoded_texts = []
		for entry in data:
			instruction_plus_input = format_input(entry)
			response_text = f"\n\n### Response:\n{entry['output']}"
			full_text = instruction_plus_input + response_text
			self.encoded_texts.append(tokenizer.encode(full_text))

	def __len__(self):
		"""
		返回数据集中条目的数量。

		Returns:
			int: 数据集中的条目数量。
		"""
		return len(self.data)

	def __getitem__(self, index):
		"""
		根据索引返回编码后的文本。

		Args:
			index (int): 要检索的条目的索引。

		Returns:
			torch.Tensor: 编码后的文本张量。
		"""
		return self.encoded_texts[index]

def custom_collate_draft_1(batch, pad_token_id = 50256, device = "cpu"):
	"""
	自定义collate函数，用于对批次数据进行填充和堆叠。

	Args:
		batch (list): 包含编码文本的批次数据列表。
		pad_token_id (int, optional): 用于填充的token ID。默认为50256。
		device (str, optional): 将张量移动到的设备（例如"cpu"或"cuda"）。默认为"cpu"。

	Returns:
		torch.Tensor: 填充并堆叠后的输入张量。
	"""
	batch_max_length = max(len(item) + 1 for item in batch)
	inputs_lst = []

	for item in batch:
		new_item = item.copy()
		new_item += [pad_token_id]
		
		padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
		inputs = torch.tensor(padded[:-1])
		inputs_lst.append(inputs)

	inputs_tensor = torch.stack(inputs_lst).to(device)
	return inputs_tensor

if __name__ == "__main__":
	tokenizer = tiktoken.get_encoding("gpt2")
	file_path = "C:/Users/13106/Desktop/LLM-From-Scratch/dataset/instruction-data.json"
	url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_mainchapter-code/instruction-data.json"

	path = "C:/Users/13106/Desktop/LLM-From-Scratch/dataset/instruction-data.json"
	with open(path, 'r', encoding='utf-8') as f:
		data = json.load(f)
	model_input = format_input(data[50])
	desired_response = f"\n\n### Response:\n{data[50]['output']}"
	print(model_input + desired_response)

	train_portion = int(len(data) * 0.85)
	test_portion = int(len(data) * 0.1)
	val_portion = int(data) - train_portion - test_portion

	train_data = data[:train_portion]
	test_data = data[train_portion:train_portion + test_portion]
	val_data = data[train_portion + test_portion:]
