import json
import os
import urllib.request
import torch
from torch.utils.data import DataLoader, Dataset
import tiktoken
from functools import partial
from GPTmodel import GPTmodel
from pytorch_dataset import load_weights_into_gpt
from gpt_download import download_and_load_gpt2
from temperature_scaling import generate
from generate_text import text_to_ids, ids_to_text
from train_model import calc_loss_loader, train_model_simple
import time



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

def custom_collate_draft_fn(batch, pad_token_id=50256, ignore_index=-100, allow_max_length=None, device="cpu"):
    """
    自定义collate函数，用于对批次数据进行填充、堆叠，并处理目标序列中的填充部分。
    此版本经过优化，修正了掩码逻辑并提高了截断效率。

    Args:
        batch (list): 包含编码文本的批次数据列表。
        pad_token_id (int, optional): 用于填充的token ID。默认为50256。
        ignore_index (int, optional): 在计算损失时要忽略的索引。默认为-100。
        allow_max_length (int, optional): 允许的最大序列长度。如果设置，序列将被截断。默认为None。
        device (str, optional): 将张量移动到的设备（例如"cpu"或"cuda"）。默认为"cpu"。

    Returns:
        tuple: 包含填充并堆叠后的输入张量和目标张量的元组。
    """
    # 1. 如果指定了最大长度，首先对所有序列进行截断
    if allow_max_length is not None:
        batch = [item[:allow_max_length] for item in batch]

    # 2. 计算批次中所有序列的最大长度
    batch_max_length = max(len(item) for item in batch)
    
    inputs_list, targets_list = [], []

    for item in batch:
        # 3. 准备输入和目标序列
        # 输入是原始序列，目标是向左移动一位的序列
        input_ids = item
        target_ids = item[1:]

        # 4. 对输入和目标进行填充
        # 输入序列填充到 batch_max_length
        pad_len_input = batch_max_length - len(input_ids)
        inputs = torch.tensor(input_ids + [pad_token_id] * pad_len_input)

        # 目标序列填充到 batch_max_length - 1，并添加一个结尾token
        pad_len_target = (batch_max_length - 1) - len(target_ids)
        targets = torch.tensor(target_ids + [pad_token_id] * pad_len_target)

        # 5. 创建掩码，将所有填充位置在目标中设置为 ignore_index
        # 这是为了在计算损失时忽略它们
        mask = targets == pad_token_id
        targets[mask] = ignore_index
        
        inputs_list.append(inputs)
        targets_list.append(targets)

    # 6. 将列表堆叠成张量并移动到指定设备
    # 注意：这里我们创建的inputs和targets长度是一致的
    # 模型结构中，输入序列的最后一个token的输出会对应目标的最后一个token
    # 但由于目标中填充位被忽略，所以这是正确的
    inputs_tensor = torch.stack(inputs_list).to(device)
    targets_tensor = torch.stack(targets_list).to(device)
    
    # 为了匹配模型输入，我们需要确保输入和目标的长度相同
    # 模型的输入是 `inputs_tensor`，期望的输出是 `targets_tensor`
    # 在典型的自回归模型中，`model(inputs_tensor)` 的输出形状会与 `inputs_tensor` 相同
    # 因此，我们需要调整 `targets_tensor` 来匹配
    final_targets = torch.full_like(inputs_tensor, ignore_index)
    final_targets[:, :-1] = targets_tensor
    
    return inputs_tensor, final_targets


if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	cunstomized_collate_fn = partial(custom_collate_draft_fn, device=device, allow_max_length = 1024)
	
	num_workers = 0
	batch_size = 8
	torch.manual_seed(123)
	tokenizer = tiktoken.get_encoding("gpt2")

	path = "C:/Users/13106/Desktop/LLM-From-Scratch/dataset/instruction-data.json"
	with open(path, 'r', encoding='utf-8') as f:
		data = json.load(f)
	desired_response = f"\n\n### Response:\n{data[50]['output']}"

	train_portion = int(len(data) * 0.85)
	test_portion = int(len(data) * 0.1)
	val_portion = len(data) - train_portion - test_portion

	train_data = data[:train_portion]
	test_data = data[train_portion:train_portion + test_portion]
	val_data = data[train_portion + test_portion:]

	train_dataset = InstructionDataset(train_data, tokenizer)
	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		collate_fn=cunstomized_collate_fn,
		shuffle=True,
		drop_last=True,
		num_workers=num_workers
	)

	val_dataset = InstructionDataset(val_data, tokenizer)
	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		collate_fn=cunstomized_collate_fn,
		shuffle=False,
		drop_last=False,
		num_workers=num_workers
	)

	test_dataset = InstructionDataset(test_data, tokenizer)
	test_loader = DataLoader(
		test_dataset,
		batch_size=batch_size,
		collate_fn=cunstomized_collate_fn,
		shuffle=False,
		drop_last=False,
		num_workers=num_workers
	)

	BASE_CONFIG = {
		"vocab_size": 50257, # Vocabulary size
		"context_length": 1024, # Context length
		"drop_rate": 0.0, # Dropout rate
		"qkv_bias": True # Query-key-value bias
	}

	model_configs = {
		"gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
		"gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
		"gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
		"gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
	}	

	CHOOSE_MODEL = "gpt2-medium (355M)"
	BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

	model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
	settings, params = download_and_load_gpt2(model_size = model_size, models_dir= r"D:\Program Files (x86)\weight")
	model = GPTmodel(BASE_CONFIG)
	load_weights_into_gpt(model, params)

	model.to(device)
	with torch.no_grad():
		train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
		val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
	print("Training Loss:", train_loss)
	print("Validation Loss:", val_loss)

	start_time = time.time()
	optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
	num_epochs = 2

	train_losses, val_losses, track_tokens_seen = train_model_simple(
		model, train_loader, val_loader,optimizer, device,
		num_epochs=num_epochs, eval_iter= 5, start_context= format_input(val_data[0]), tokenizer=tokenizer
	)

	end_time = time.time()
	execution_time = (end_time - start_time) / 60
	print(f"Training completed in {execution_time:.2f} minutes.")
	