import torch
import torch.nn as nn
import tiktoken
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from .temperature_scaling import generate

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 将项目根目录添加到sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from .GPTmodel import GPTmodel, generate_text_simple
from .GPT_dataset_V1 import create_dataloader_v1
from .utils import load_text_data
from .generate_text import text_to_ids, ids_to_text

# 移入的函数：损失计算
def cal_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = cal_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

# 评估模型在训练集和验证集上的性能
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        if len(val_loader) == 0:
            val_loss = 0.0
        else:
            val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

# 生成并打印文本样本
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    # 当使用 DataParallel 包装模型时，需要通过 .module 访问原始模型属性
    if isinstance(model, nn.DataParallel):
        context_size = model.module.pos_emb.weight.shape[0]
    else:
        context_size = model.pos_emb.weight.shape[0]
    
    encoded = torch.tensor(text_to_ids(start_context, tokenizer)).unsqueeze(0).to(device)
    with torch.no_grad():
        # 使用简单的生成函数进行快速、确定性的采样
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )
        decoded_text = ids_to_text(token_ids, tokenizer)
        logging.info(f"Sample: {decoded_text.replace("\n", " ")}")
    model.train()

# 简单的模型训练函数
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_loss, val_loss, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = cal_loss_batch(input_batch, target_batch, model, device)
            # 当使用 DataParallel 时，损失是一个包含在每个GPU上的损失的张量。
            # .mean() 将它们聚合起来用于反向传播。
            if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss_val, val_loss_val = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_loss.append(train_loss_val)
                val_loss.append(val_loss_val)
                track_tokens_seen.append(tokens_seen)
                logging.info(f"Epoch {epoch + 1} (Step {global_step:06d}): "
                             f"Train Loss {train_loss_val:.3f}, Val Loss {val_loss_val:.3f}")
        
        generate_and_print_sample(model, tokenizer, device, start_context)
    
    plot_losses(track_tokens_seen, train_loss, val_loss)
    return train_loss, val_loss, track_tokens_seen

# 新增绘图函数
def plot_losses(epochs_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs_seen, train_losses, label="Train Loss")
    ax1.plot(epochs_seen, val_losses, label="Validation Loss")
    ax1.set_xlabel("Tokens Seen")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss_plot.png")
    logging.info("Loss plot saved to training_loss_plot.png")
    plt.close(fig)

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

if __name__ == "__main__":
    file_path = project_root / "dataset" / "the-verdict.txt"
    text_data = load_text_data(file_path)
    if text_data is None:
        logging.error("无法加载文本数据，退出程序。")
        sys.exit(1)

    train_ratio = 0.9
    split_idx = int(len(text_data) * train_ratio)
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    tokenizer = tiktoken.get_encoding("gpt2")
    train_loader = create_dataloader_v1(
        train_data,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)
    model = GPTmodel(GPT_CONFIG_124M)
    
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model, device_ids=[0, 1, 2])

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 10
    
    train_model_simple(
        model, train_loader, 
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=1,
        start_context="Every effort moves you",
        tokenizer=tokenizer
    )
    
    # 训练后，使用带有温度缩放的生成函数来获得更多样化的输出
    logging.info("\nGenerating text with temperature scaling...")
    token_ids = generate(
        model=model,
        idx=torch.tensor(text_to_ids("Every effort moves you", tokenizer)).unsqueeze(0).to(device),
        context_size=GPT_CONFIG_124M["context_length"],
        max_new_tokens=25,
        temperature=1.4
    )
    logging.info(f"Final generated text: {ids_to_text(token_ids, tokenizer)}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": GPT_CONFIG_124M
        }, 
        project_root / "models" / "gpt_model.pth"
        )
