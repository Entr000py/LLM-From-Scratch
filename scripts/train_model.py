import torch
import torch.nn as nn
import tiktoken
import sys
from pathlib import Path
import matplotlib.pyplot as plt # 新增导入

# 将项目根目录添加到sys.path，以便导入Chapter4和Chapter2中的模块
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from GPTmodel import GPTmodel, generate_text_simple
from GPT_dataset_V1 import create_dataloader_v1
from generate_text import text_to_ids, ids_to_text, batch_ids_to_text, cal_loss_batch, calc_loss_loader
# 评估模型在训练集和验证集上的性能
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 在此上下文中禁用梯度计算
        # 计算训练损失
        train_loss = calc_loss_loader(train_loader, model, device, num_batches = eval_iter)
        # 计算验证损失
        if len(val_loader) == 0: # 检查val_loader是否为空
            val_loss = 0.0 # 如果为空，则将val_loss设置为0.0
        else: 
            val_loss = calc_loss_loader(val_loader, model, device, num_batches = eval_iter)
    model.train()  # 将模型设置回训练模式
    return train_loss, val_loss  # 返回训练损失和验证损失

# 生成并打印文本样本
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()  # 将模型设置为评估模式
    # 获取模型的位置嵌入大小，即上下文窗口大小
    context_size = model.pos_emb.weight.shape[0]
    # 将起始上下文文本编码为token ID，并移动到指定设备
    encoded = text_to_ids(start_context, tokenizer).unsqueeze(0).to(device)
    with torch.no_grad():  # 在此上下文中禁用梯度计算
        # 使用模型生成文本token ID
        token_ids = generate_text_simple(
            model = model,
            idx = encoded,
            max_new_tokens = 50,  # 生成的最大新token数量
            context_size = context_size
        )
        # 将生成的token ID解码为文本
        decoded_text = batch_ids_to_text(token_ids, tokenizer)
        # 打印解码后的文本，将换行符替换为空格
        print(decoded_text.replace("\n", " "))
    model.train()  # 将模型设置回训练模式

# 简单的模型训练函数
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    # 初始化用于跟踪训练过程的列表：训练损失、验证损失和已处理的token数量
    train_loss, val_loss, track_tokens_seen = [], [], []
    # 初始化已处理的token数量和全局步数
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):  # 遍历每个训练周期
        model.train()  # 将模型设置为训练模式
        for input_batch, target_batch in train_loader:  # 遍历训练数据加载器中的每个批次
            optimizer.zero_grad()  # 清除之前的梯度
            # 计算当前批次的损失
            loss = cal_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数
            # 累计已处理的token数量
            tokens_seen += input_batch.numel()
            global_step += 1  # 增加全局步数

            # 每隔eval_freq步评估一次模型
            if global_step % eval_freq == 0:
                # 评估模型在训练集和验证集上的表现
                train_loss_val, val_loss_val = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                # 记录损失和已处理的token数量
                train_loss.append(train_loss_val)
                val_loss.append(val_loss_val)
                track_tokens_seen.append(tokens_seen)
                # 打印当前训练状态
                print(f"Epoch {epoch + 1} (Step {global_step:06d}): "
                      f"Train Loss {train_loss_val:.3f}, Val Loss {val_loss_val:.3f}")
        
        # 每个epoch结束后生成并打印样本文本
        generate_and_print_sample(model, tokenizer, device, start_context)
    
    # 在训练结束后绘制损失曲线
    plot_losses(track_tokens_seen, train_loss, val_loss) # 新增调用

    # 返回训练过程中记录的数据：训练损失列表、验证损失列表和已处理token数量列表
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
    plt.savefig("training_loss_plot.png") # 保存图像
    plt.close(fig) # 关闭图形，释放内存

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

def load_text_data(file_path):
    """Load text data from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text_data = file.read()
        return text_data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

if __name__ == "__main__":
    # Load text data from file using project root
    file_path = project_root / "dataset" / "the-verdict.txt"
    text_data = load_text_data(file_path)
    if text_data is None:
        print("无法加载文本数据，退出程序。")
        sys.exit(1) # 退出程序

    train_ratio = 0.9
    split_idx = int(len(text_data) * train_ratio)
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    tokenizer = tiktoken.get_encoding("gpt2")
    train_loader =  create_dataloader_v1(
        train_data,
        tokenizer= tokenizer,
        batch_size= 2,
        max_length= GPT_CONFIG_124M["context_length"],
        stride= GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers= 0
    )

    val_loader = create_dataloader_v1(
        val_data,
        tokenizer= tokenizer,
        batch_size= 2,
        max_length= GPT_CONFIG_124M["context_length"],
        stride= GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers= 0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)  # 设置随机种子以确保结果可重复
    model = GPTmodel(GPT_CONFIG_124M)
    model.to(device)  # 将模型移动到指定设备
    optimizer = torch.optim.AdamW(model.parameters(), lr  = 0.0004, weight_decay= 0.1)  # 创建优化器
    num_epochs = 10
    train_loss, val_loss, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs = num_epochs, eval_freq = 5, eval_iter= 1, start_context= "Every effort moves you", tokenizer = tokenizer
        )
