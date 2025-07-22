import torch
import tiktoken
import sys
from pathlib import Path

# Add the project root to sys.path to allow importing modules from Chapter4
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from GPTmodel import GPTmodel, generate_text_simple
from GPT_dataset_V1 import create_dataloader_v1

def text_to_token_ids(text, tokenizer):
    # 将文本转换为token ID
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    # 将token ID转换回文本
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

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

def cal_loss_batch(input_batch, target_batch, model, device):
    """
    Calculate the loss for a single batch.
    
    Args:
        input_batch: Input tensor batch
        target_batch: Target tensor batch
        model: The model to evaluate
        device: The device to run calculations on
        
    Returns:
        torch.Tensor: The calculated loss
    """
    input_batch, target_batch =input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches = None):
    """
    Calculate the average loss across all batches in a data loader.
    
    Args:
        data_loader: DataLoader containing batches
        model: The model to evaluate
        device: The device to run calculations on
        num_batches: Optional number of batches to process
        
    Returns:
        float: The average loss across all processed batches
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else :
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = cal_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    
    return total_loss / num_batches

if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 256,  # Context length
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-Key-Value bias
    }

    torch.manual_seed(123)
    model = GPTmodel(GPT_CONFIG_124M)
    model.eval()

    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    # 生成文本
    token_ids = generate_text_simple(
        model = model,
        idx = text_to_token_ids(start_context, tokenizer),
        max_new_tokens = 10,
        contex_size = GPT_CONFIG_124M["context_length"]
    )

    # 打印生成的文本
    #print("Output text:", token_ids_to_text(token_ids, tokenizer))

    inputs = torch.tensor([[16833, 3626, 6100],  # ["every effort moves",
                        [40, 1107, 588]])     # "I really like"]

    targets = torch.tensor([[3626, 6100, 345],   # [" effort moves you",
                            [107, 588, 11311]])  # " really like chocolate"]

    # 获取模型输出的logits
    logits = model(inputs)
    # 将logits转换为概率分布
    probas = torch.softmax(logits, dim=-1)
    # 打印概率张量的形状：[2, 3, 50257]
    print(probas.shape)
    # 选择概率最高的token
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    # 打印生成的token IDs
    print("Token ids:\n", token_ids)
    # 打印目标文本
    print("Target batch 1:", {token_ids_to_text(targets[0], tokenizer)})
    # 打印生成的文本
    print("Output batch 1:", {token_ids_to_text(token_ids[0].flatten(), tokenizer)})

    # 目标token与初始softmax概率得分
    text_idx = 0
    # 获取目标token的概率
    target_probs_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text1 :", target_probs_1)

    text_idx = 1
    # 获取目标token的概率
    target_probs_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text2 :", target_probs_2)

    # 计算对数概率
    log_probas = torch.log(torch.cat((target_probs_1, target_probs_2)))
    print("log probas:", log_probas)

    # 计算平均对数概率
    avg_log_probas = torch.mean(log_probas)
    print("Average log probas:", avg_log_probas)

    # 计算负平均对数概率
    neg_avg_log_probas = avg_log_probas * -1
    print("neg_avg_log_probas:", neg_avg_log_probas)

    print("Logits shape :", logits.shape)
    print("Targets shape :", targets.shape)

    logits_flat = logits.flatten(0, 1)
    target_flat = targets.flatten()
    print("Logits flat shape :", logits_flat.shape)
    print("Targets flat shape :", target_flat.shape)

    loss = torch.nn.functional.cross_entropy(logits_flat, target_flat)
    print("Loss :", loss)

    # Load text data from file using project root
    file_path = project_root / "dataset" / "the-verdict.txt"
    # Load the text data
    text_data = load_text_data(file_path)

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

    print("\nTrain loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    print(f"Train loss: {train_loss:.4f}, \nVal loss: {val_loss:.4f}")
