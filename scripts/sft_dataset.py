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
	# 计算批次中所有序列的最大长度。
	# 这里的 +1 是为了在每个序列末尾添加一个额外的pad_token_id，作为序列结束标记或用于后续处理。
	batch_max_length = max(len(item) + 1 for item in batch)
	inputs_lst = []

	# 遍历批次中的每个编码文本序列
	for item in batch:
		new_item = item.copy() # 创建当前序列的副本，避免修改原始数据
		new_item += [pad_token_id] # 在序列末尾添加一个pad_token_id，作为序列结束标记

		# 对序列进行填充，使其达到批次的最大长度。
		# 使用pad_token_id进行填充。
		padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
		
		# 将填充后的序列转换为PyTorch张量。
		inputs = torch.tensor(padded[:-1])
		inputs_lst.append(inputs) # 将处理后的张量添加到列表中

	# 将所有填充后的输入张量堆叠成一个单一的张量，并将其移动到指定的设备（CPU或CUDA）
	inputs_tensor = torch.stack(inputs_lst).to(device)
	return inputs_tensor

def custom_collate_draft_2(batch, pad_token_id = 50256, device = "cpu"):
	"""
	自定义collate函数，用于对批次数据进行填充和堆叠，并生成输入和目标张量。

	Args:
		batch (list): 包含编码文本的批次数据列表。
		pad_token_id (int, optional): 用于填充的token ID。默认为50256。
		device (str, optional): 将张量移动到的设备（例如"cpu"或"cuda"）。默认为"cpu"。

	Returns:
		tuple: 包含填充并堆叠后的输入张量和目标张量的元组。
	"""
	# 计算批次中所有序列的最大长度。
	# 这里的 +1 是为了在每个序列末尾添加一个额外的pad_token_id，作为序列结束标记或用于后续处理。
	batch_max_length = max(len(item) + 1 for item in batch)
	inputs_lst, targets_lst = [], []

	for item in batch:
		new_item = item.copy()
		new_item += [pad_token_id] # 在序列末尾添加一个pad_token_id，作为序列结束标记

		# 对序列进行填充，使其达到批次的最大长度。使用pad_token_id进行填充。
		padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
		#[:-1]: 表示从列表的开头到倒数第二个元素（不包括最后一个元素）
		inputs = torch.tensor(padded[:-1])
		# targets[1:] 表示目标序列从第二个token开始，是输入的下一个token。
		targets = torch.tensor(padded[1:])
		
		inputs_lst.append(inputs) # 将处理后的输入张量添加到列表中
		targets_lst.append(targets) # 将处理后的目标张量添加到列表中

	inputs_tensor = torch.stack(inputs_lst).to(device)
	targets_tensor = torch.stack(targets_lst).to(device)

	return inputs_tensor, targets_tensor # 返回最终的输入和目标张量

def custom_collate_draft_fn(batch, pad_token_id = 50256, ignore_index = -100, allow_max_length = None, device = "cpu"):
	"""
	自定义collate函数，用于对批次数据进行填充、堆叠，并处理目标序列中的填充部分。

	Args:
		batch (list): 包含编码文本的批次数据列表。
		pad_token_id (int, optional): 用于填充的token ID。默认为50256。
		ignore_index (int, optional): 在计算损失时要忽略的索引。默认为100。
		allow_max_length (int, optional): 允许的最大序列长度。如果设置，序列将被截断。默认为None。
		device (str, optional): 将张量移动到的设备（例如"cpu"或"cuda"）。默认为"cpu"。

	Returns:
		tuple: 包含填充并堆叠后的输入张量和目标张量的元组。
	"""
	# 这里的 +1 是为了在每个序列末尾添加一个额外的pad_token_id，作为序列结束标记或用于后续处理。
	batch_max_length = max(len(item) + 1 for item in batch)
	inputs_list, targets_list = [], []

	for item in batch:
		new_item = item.copy()
		new_item += [pad_token_id] # 在序列末尾添加一个pad_token_id，作为序列结束标记

		# 使用pad_token_id进行填充。
		padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
		
		# inputs[:-1] 表示输入序列是原始序列除了最后一个token。
		inputs = torch.tensor(padded[:-1])
		# targets[1:] 表示目标序列从第二个token开始，是输入的下一个token。
		targets = torch.tensor(padded[1:])

		# 创建一个掩码，标记目标序列中等于pad_token_id的位置。
		mask = targets == pad_token_id
		# 找到掩码中非零（即为True）的索引，并移除维度为1的维度。
		indices = torch.nonzero(mask).squeeze()
		# 如果找到的填充索引数量大于1（即存在多个填充token），
		# 则将从第二个填充token开始的所有填充token的索引设置为ignore_index。
		# 这样做是为了在计算损失时忽略这些填充token，避免它们对模型训练产生影响。
		if indices.numel() > 1:
			targets[indices[1:]] = ignore_index

		if allow_max_length is not None:
			inputs = inputs[:allow_max_length]
			targets = targets[:allow_max_length]
		
		inputs_list.append(inputs)
		targets_list.append(targets)

	inputs_tensor = torch.stack(inputs_list).to(device)
	targets_tensor = torch.stack(targets_list).to(device)

	return inputs_tensor, targets_tensor


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
	val_portion = len(data) - train_portion - test_portion

	train_data = data[:train_portion]
	test_data = data[train_portion:train_portion + test_portion]
	val_data = data[train_portion + test_portion:]

	inputs_1 = [0, 1, 2, 3, 4]
	inputs_2 = [5, 6]
	inputs_3 = [7, 8, 9]
	batch = (
		inputs_1,
		inputs_2,
		inputs_3
	)
	# print(custom_collate_draft_1(batch))

	inputs, targets = custom_collate_draft_fn(batch)
	print(inputs)
	print(targets)
