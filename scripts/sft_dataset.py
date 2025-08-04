import json
import os
import urllib.request
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import tiktoken
from functools import partial
from GPTmodel import GPTmodel,replace_liner_with_lora
from pytorch_dataset import load_weights_into_gpt
from gpt_download import download_and_load_gpt2
from temperature_scaling import generate
from generate_text import text_to_ids, ids_to_text
from train_model import calc_loss_loader, train_model_simple
import torch.optim.lr_scheduler as lr_scheduler
import time
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup # 导入带预热的余弦退火调度器


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
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

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
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allow_max_length is not None:
            inputs = inputs[:allow_max_length]
            targets = targets[:allow_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    customized_collate_fn = partial(custom_collate_fn, device=device, allow_max_length = 1024)
	
    num_workers = 0
    batch_size = 8
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
        "drop_rate": 0.1, # Dropout rate
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
    print("BASE_CONFIG before GPTmodel initialization:", BASE_CONFIG) # Added for debugging
    model = GPTmodel(BASE_CONFIG)
    load_weights_into_gpt(model, params)

    #LoRA
    # replace_liner_with_lora(model, rank = 16, alpha = 32)

    model.to(device)
    # print("Model Information:", model)

    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 3
	
    # 初始化学习率调度器
    total_steps = num_epochs * len(train_loader)  # 计算总步数
    num_warmup_steps = int(0.1 * total_steps) # 预热步数设置为总步数的10%
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps 
    )

    train_losses, val_losses, _ = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_iter=5, start_context=format_input(val_data[0]), tokenizer=tokenizer, eval_freq=5,
        scheduler=scheduler,
        patience=7, min_delta=0.001,
        max_grad_norm=1.0 # 添加梯度裁剪参数
    )

    end_time = time.time()
    execution_time = (end_time - start_time) / 60
    print(f"Training completed in {execution_time:.2f} minutes.")

    # 添加交互式文本生成功能
    # print("\n模型训练已完成。现在可以输入文本与模型交互。")
    # print("输入 'exit' 退出程序。")
    
    # while True:
    #     try:
    #         input_text = input("\n请输入您的文本 (输入 'exit' 退出): ")
    #         if input_text.lower() == 'exit':
    #             print("退出程序。")
    #             break
            
    #         # 将输入文本转换为token ID
    #         token_ids = text_to_ids(input_text, tokenizer)
    #         input_tensor = torch.tensor(token_ids).unsqueeze(0).to(device)
            
    #         # 使用模型生成文本
    #         generated_token_ids = generate(
    #             model=model,
    #             idx=input_tensor,
    #             max_new_tokens=256,
    #             context_size=BASE_CONFIG["context_length"],
    #             top_p=0.9,
    #             repetition_penalty=1.2,
    #             eos_id=50256
    #         )
            
    #         # 将生成的token ID转换回文本
    #         generated_text = ids_to_text(generated_token_ids, tokenizer)
            
    #         # 打印生成的文本（去除输入部分）
    #         response_text = generated_text[len(input_text):].strip()
    #         print(f"\n模型输出: {response_text}")
            
    #     except KeyboardInterrupt:
    #         print("\n\n程序被用户中断。")
    #         break
    #     except Exception as e:
    #         print(f"\n发生错误: {e}")
    #         print("请重试。")

    # for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    #     input_text = format_input(entry)
    #     token_ids = generate(
    #             model = model,
    #             idx = torch.tensor(text_to_ids(input_text, tokenizer)).unsqueeze(0).to(device),
    #             max_new_tokens= 256,
    #             context_size = BASE_CONFIG["context_length"],
    #             top_p=0.9,
    #             repetition_penalty=1.2,
    #             eos_id= 50256
    #     )
    #     generated_text_str = ids_to_text(token_ids, tokenizer)
    #     response_text = generated_text_str[len(input_text):].replace("### Response:",
    # "").strip()
    #     test_data[i]["model_response"] = response_text

    # with open("instruction-data-with-response.json", "w") as file:
    #     json.dump(test_data, file, indent=4)
