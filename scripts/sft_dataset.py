import json
import os
import urllib.request
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import tiktoken
from functools import partial
from GPTmodel import GPTmodel
from pytorch_dataset import load_weights_into_gpt
from gpt_download import download_and_load_gpt2
from temperature_scaling import generate
from generate_text import text_to_ids, ids_to_text
from train_model import calc_loss_loader, train_model_simple
import torch.optim.lr_scheduler as lr_scheduler  # 导入学习率调度器
import time
from tqdm import tqdm


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
		"Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
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
		self.prompt_lens = []
		# 获取EOS token的ID，对于GPT-2是50256
		eos_token_id = tokenizer.eot_token
		for entry in data:
			instruction_plus_input = format_input(entry)
			response_text = f"\n\n### Response:\n{entry['output']}"
			full_text = instruction_plus_input + response_text
			
			# 编码文本并手动添加EOS token
			encoded = tokenizer.encode(full_text)
			encoded.append(eos_token_id)
			self.encoded_texts.append(encoded)

			# 计算并存储prompt_len
			prompt_encoded = tokenizer.encode(instruction_plus_input)
			self.prompt_lens.append(len(prompt_encoded))
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
			dict: 包含 'encoded' (torch.Tensor) 和 'prompt_len' (int) 的字典。
		"""
		return {
			'encoded': torch.tensor(self.encoded_texts[index], dtype=torch.long),
			'prompt_len': self.prompt_lens[index]
		}

def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, device="cpu", allow_max_length=None):
    """
    自定义的collate函数，用于对批次数据进行填充和处理，为模型训练准备输入和目标。
    这个版本是经过修正的，可以正确处理目标序列和损失掩码。

    Args:
        batch (list of dict or list of list): 包含 'encoded' (list of int) 和 'prompt_len' (int) 的批次数据。
                                              如果是列表，则应为 [encoded_tensor, prompt_len] 的形式。
        pad_token_id (int): 用于填充的token ID。
        ignore_index (int): 在计算损失时要忽略的索引。
        device (str): 将张量移动到的设备（"cpu"或"cuda"）。
        allow_max_length (int, optional): 允许的最大序列长度。如果设置，序列将被截断。

    Returns:
        tuple: 包含输入张量 (inputs) 和目标张量 (targets) 的元组。
    """
    # 从批次中提取编码文本和提示长度
    # 检查 batch 中的 item 是字典还是列表/元组
    if isinstance(batch[0], dict):
        encoded_items = [item['encoded'] for item in batch]
        prompt_lens = [item['prompt_len'] for item in batch]
    else:
        # 假设 item 是一个列表或元组，形式为 [encoded_tensor, prompt_len]
        encoded_items = [item[0] for item in batch]
        prompt_lens = [item[1] for item in batch]

    # 1. 如果指定了最大长度，首先对所有序列进行截断
    if allow_max_length is not None:
        encoded_items = [item[:allow_max_length] for item in encoded_items]

    # 2. 使用 pad_sequence 对所有序列进行填充，使它们达到相同的长度
    # pad_sequence 需要序列是 Tensor 列表，并且填充到批次中最长序列的长度
    padded_tensor = pad_sequence(encoded_items, batch_first=True, padding_value=pad_token_id)

    # 4. 创建输入和目标
    inputs = padded_tensor[:, :-1].contiguous()
    targets = padded_tensor[:, 1:].contiguous()

    # 5. 创建损失掩码
    for i, prompt_len in enumerate(prompt_lens):
        # 确保掩码长度不超过目标张量的实际长度
        mask_len = min(prompt_len - 1, inputs.shape[1])
        if mask_len > 0:
            targets[i, :mask_len] = ignore_index
    
    # 屏蔽所有填充部分的token
    targets[targets == pad_token_id] = ignore_index

    # 6. 将张量移动到指定设备
    return inputs.to(device), targets.to(device)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    customized_collate_fn = partial(custom_collate_fn, device=device, allow_max_length = 1024)
	
    num_workers = 0
    batch_size = 4
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")

    path = r"/storage/jiangfei/LLM-From-Scratch/dataset/instruction-data.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

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
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
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
    settings, params = download_and_load_gpt2(model_size = model_size, models_dir= r"/storage/jiangfei/LLM-From-Scratch/weight")
    model = GPTmodel(BASE_CONFIG)
    load_weights_into_gpt(model, params)

    model.to(device)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    print("Training Loss:", train_loss)
    print("Validation Loss:", val_loss)

    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.1)
    num_epochs = 2
	
    # 初始化学习率调度器
    total_steps = num_epochs * len(train_loader)  # 计算总步数
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    train_losses, val_losses, _ = train_model_simple( # 移除未使用的 track_tokens_seen
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_iter=5, start_context=format_input(val_data[0]), tokenizer=tokenizer, eval_freq=5,
        scheduler=scheduler,  # 传递调度器
        patience=3, min_delta=0.001  # 添加早停参数
    )

    end_time = time.time()
    execution_time = (end_time - start_time) / 60
    print(f"Training completed in {execution_time:.2f} minutes.")

    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)
        token_ids = generate(
                model = model,
                idx = torch.tensor(text_to_ids(input_text, tokenizer)).unsqueeze(0).to(device),
                max_new_tokens= 256,
                context_size = BASE_CONFIG["context_length"],
                top_p=0.9,
                repetition_penalty=1.2,
                eos_id= 50256
        )
        generated_text_str = ids_to_text(token_ids, tokenizer)
        response_text = generated_text_str[len(input_text):].replace("### Response:",
    "").strip()
        test_data[i]["model_response"] = response_text

    with open("instruction-data-with-response.json", "w") as file:
        json.dump(test_data, file, indent=4)
