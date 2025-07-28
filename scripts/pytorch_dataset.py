import pandas as pd
from tensorflow.python.ops.resource_variable_ops import variable_accessed
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from GELU import FeedForward
from gpt_download import download_and_load_gpt2
from GPTmodel import GPTmodel, generate_text_simple
from transformer import TransformerBlock
from multi_head_attention import MultiHeadAttention, MultiHeadAttentionWrapper
from generate_text import text_to_ids, ids_to_text
import numpy as np
import os
import logging

def assign(left, right):
    """
    将右侧 Tensor 的内容复制到左侧 Tensor 中。
    如果形状不匹配，则抛出 ValueError。

    Args:
        left (torch.Tensor): 目标 Tensor，内容将被覆盖。
        right (numpy.ndarray or torch.Tensor): 源数据，其内容将被复制。

    Returns:
        torch.Tensor: 更新后的左侧 Tensor。

    Raises:
        ValueError: 如果 left 和 right 的形状不匹配。
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    with torch.no_grad():
        left.copy_(torch.tensor(right))
    return left

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])               #A
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    for b in range(len(params["blocks"])):                                       #B
        q_w, k_w, v_w = np.split(                                                #C
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.weight = assign(
            gpt.trf_blocks[b].norm1.weight,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.bias = assign(
            gpt.trf_blocks[b].norm1.bias,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.weight = assign(
            gpt.trf_blocks[b].norm2.weight,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.bias = assign(
            gpt.trf_blocks[b].norm2.bias,
            params["blocks"][b]["ln_2"]["b"])
    
    gpt.final_norm.weight = assign(gpt.final_norm.weight, params["g"])
    gpt.final_norm.bias = assign(gpt.final_norm.bias, params["b"])

class SpamDataset(Dataset):
    """
    用于垃圾邮件分类的 PyTorch Dataset。
    处理 CSV 文件中的文本数据，并将其编码为模型输入。
    """
    def __init__(self, csv_files, tokenizer, max_length=None, pad_token_id=50256):
        """
        初始化 SpamDataset。

        Args:
            csv_files (str): 包含文本和标签的 CSV 文件路径。
            tokenizer: 用于编码文本的 tokenizer 对象。
            max_length (int, optional): 编码文本的最大长度。如果为 None，则使用数据集中最长编码文本的长度。
            pad_token_id (int, optional): 用于填充短序列的 token ID。默认为 50256。
        """
        self.data = pd.read_csv(csv_files)

        # 使用 tokenizer 对所有文本进行编码
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data['text']
        ]

        if max_length is None:
            # 如果未指定 max_length，则使用数据集中最长编码文本的长度
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # 如果指定了 max_length，则截断过长的序列
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]
        # 填充短序列到 max_length
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
        
    def __getitem__(self, index):
        """
        根据索引获取编码文本和对应的标签。

        Args:
            index (int): 数据点的索引。

        Returns:
            tuple: 包含编码文本的 Tensor 和标签的 Tensor。
        """
        encoded_text = self.encoded_texts[index]
        label = self.data.iloc[index]['label']
        return (
            torch.tensor(encoded_text, dtype=torch.long), # 将编码文本转换为 Long Tensor
            torch.tensor(label, dtype=torch.long) # 将标签转换为 Long Tensor
        )

    def __len__(self):
        """
        返回数据集中的样本数量。

        Returns:
            int: 数据集中的样本数量。
        """
        return len(self.data)
    
    def _longest_encoded_length(self):
        """
        计算数据集中最长编码文本的长度。

        Returns:
            int: 最长编码文本的长度。
        """
        return max(len(encoded_text) for encoded_text in self.encoded_texts)

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """
    计算模型在给定数据加载器上的准确率。

    Args:
        data_loader (DataLoader): 用于加载数据的 PyTorch DataLoader。
        model (nn.Module): 要评估的 PyTorch 模型。
        device (torch.device): 模型和数据所在的设备（例如 'cpu' 或 'cuda'）。
        num_batches (int, optional): 要处理的批次数量。如果为 None，则处理所有批次。

    Returns:
        float: 模型的准确率。
    """
    model.eval()  # 将模型设置为评估模式
    correct_predictions, num_examples = 0, 0  # 初始化正确预测数和样本总数
    
    if num_batches is None:
        num_batches = len(data_loader)  # 如果未指定批次数量，则使用数据加载器中的所有批次
    else:
        num_batches = min(num_batches, len(data_loader))  # 否则，使用指定数量或数据加载器中的最小批次数量

    for i, (input_batch, target_batch) in enumerate(data_loader):  # 遍历数据加载器
        if i < num_batches:  # 检查是否达到指定的批次数量
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 将输入和目标数据移动到指定设备
            
            with torch.no_grad():  # 在此块中禁用梯度计算，以节省内存和加快计算
                logits = model(input_batch)[:, -1, :]  # 通过模型获取 logits，并选择最后一个时间步的输出
            predicted_labels = torch.argmax(logits, dim = -1)  # 获取预测标签

            num_examples += target_batch.shape[0]  # 累加样本总数
            correct_predictions += (predicted_labels == target_batch).sum().item()  # 累加正确预测数
        else:
            break  # 如果达到指定的批次数量，则停止循环
    
    return correct_predictions / num_examples  # 返回准确率

def calc_loss_batch(input_batch, target_batch, model, device):
    """
    计算单个批次的交叉熵损失。

    Args:
        input_batch (torch.Tensor): 输入数据批次。
        target_batch (torch.Tensor): 目标标签批次。
        model (nn.Module): PyTorch 模型。
        device (torch.device): 模型和数据所在的设备。

    Returns:
        torch.Tensor: 计算出的损失值。
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 将输入和目标数据移动到指定设备
    logits = model(input_batch)[:, -1, :]  # 通过模型获取 logits，并选择最后一个时间步的输出
    loss = torch.nn.functional.cross_entropy(logits, target_batch)  # 计算交叉熵损失
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    计算模型在给定数据加载器上的平均损失。

    Args:
        data_loader (DataLoader): 用于加载数据的 PyTorch DataLoader。
        model (nn.Module): 要评估的 PyTorch 模型。
        device (torch.device): 模型和数据所在的设备。
        num_batches (int, optional): 要处理的批次数量。如果为 None，则处理所有批次。

    Returns:
        float: 模型的平均损失。
    """
    total_loss = 0.  # 初始化总损失
    if len(data_loader) == 0:
        return float("nan")  # 如果数据加载器为空，返回 NaN
    elif num_batches is None:
        num_batches = len(data_loader)  # 如果未指定批次数量，则使用数据加载器中的所有批次
    else:
        num_batches = min(num_batches, len(data_loader))  # 否则，使用指定数量或数据加载器中的最小批次数量
    for i, (input_batch, target_batch) in enumerate(data_loader):  # 遍历数据加载器
        if i < num_batches:  # 检查是否达到指定的批次数量
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算当前批次的损失
            total_loss += loss.item()  # 累加损失
        else:
            break  # 如果达到指定的批次数量，则停止循环
    return total_loss / num_batches  # 返回平均损失

if __name__ == '__main__':

    # 定义模型下载路径和模型名称
    download_path = r"/storage/jiangfei/LLM-From-Scratch/weight"
    model_name = "gpt2" # 或者 "gpt2-large", "gpt2-medium", "gpt2-xl"
    
    # 从预训练模型加载 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=download_path)
    
    # 设置 DataLoader 参数
    num_workers = 0
    batch_size = 8
    torch.manual_seed(123) # 设置随机种子以保证结果可复现

    # 初始化训练、验证和测试数据集
    train_dataset = SpamDataset(
        csv_files= r'/storage/jiangfei/LLM-From-Scratch/dataset/train.csv',
        max_length = None, # 自动确定最大长度
        tokenizer = tokenizer
    )
    val_dataset = SpamDataset(
        csv_files= r'/storage/jiangfei/LLM-From-Scratch/dataset/validation.csv',
        max_length = None,
        tokenizer = tokenizer
    )
    test_dataset = SpamDataset(
        csv_files= r'/storage/jiangfei/LLM-From-Scratch/dataset/test.csv',
        max_length = None,
        tokenizer = tokenizer
    )

    # 创建训练、验证和测试数据的 DataLoader
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True, # 训练集需要打乱
        num_workers = num_workers,
        drop_last = True # 丢弃最后一个不完整的批次
    )
    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size = batch_size,
        shuffle = False, # 验证集也打乱
        num_workers = num_workers,
        drop_last = True
    )
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False, # 测试集也打乱
        num_workers = num_workers,
        drop_last = True
    )
    # 遍历训练 DataLoader，此处仅为示例，实际训练中会在此处进行模型训练
    for input_batch, target_batch in train_loader:
        pass
    
    # 选择模型配置
    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves" # 文本生成时的起始提示
    BASE_CONFIG = {
        "vocab_size": 50257, # 词汇表大小
        "context_length": 1024, # 模型上下文长度
        "drop_rate": 0.0, # Dropout 比率
        "qkv_bias": True # Query-key-value 偏置
    }
    # 不同 GPT-2 模型大小的配置
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    # 更新基础配置以匹配所选模型
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    # 断言数据集的最大长度不超过模型的上下文长度
    assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
        f"Dataset length {train_dataset.max_length} exceeds model's context "
        f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
        f"`max_length={BASE_CONFIG['context_length']}`"
    )

    # 加载 Hugging Face 的 GPT2LMHeadModel
    model = GPTmodel(BASE_CONFIG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model, device_ids=[0, 1, 2])

    model.to(device)
    #冻结模型
    for param in model.parameters():
        param.requires_grad = False

    #替换为分类头
    num_classes = 2
    model.out_head = torch.nn.Linear(
        in_features=BASE_CONFIG["emb_dim"],
        out_features=num_classes
    )

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.out_head.parameters():
        param.requires_grad = True

    # 获取自定义 GPT 模型所需的参数
    settings, params = download_and_load_gpt2(model_size="124M", models_dir=download_path)
    print("Available keys in params:", params.keys())
    
    # 将下载的权重加载到模型中
    load_weights_into_gpt(model, params)

    # model.eval() # 将模型设置为评估模式

    # # 示例文本生成
    # text_1 = "Every effort moves you forward"
    # text_2 = (
    # "Is the following text 'spam'? Answer with 'yes' or 'no':"
    # " 'You are a winner you have been specially"
    # " selected to receive $1000 cash or a $2000 award.'"
    # )
    # # 将文本转换为 token ID
    # token_ids = generate_text_simple(
    #     model=model,
    #     idx=torch.tensor(text_to_ids(text_2, tokenizer)).unsqueeze(0),
    #     max_new_tokens=25,
    #     context_size=BASE_CONFIG["context_length"],
    #     temperature=0.7,
    #     top_k=50
    # )
    # # 将生成的 token ID 转换回文本并打印
    # print(ids_to_text(token_ids, tokenizer))

    #测试
    # inputs = tokenizer.encode("Do you have time")
    # inputs = torch.tensor(inputs).unsqueeze(0)
    # print("Inputs :", inputs)
    # print("Input dimension :", inputs.shape)
    # with torch.no_grad():
    #     outputs = model(inputs)
    # print("Output dimension :", outputs.shape)

    #准确率
    train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches= 10)
    var_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches= 10)
    test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches= 10)
    print(f"Train accuracy: {train_accuracy*100:.2f}")
    print(f"Validation accuracy: {var_accuracy*100:.2f}")
    print(f"Test accuracy: {test_accuracy*100:.2f}")

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
        test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)
    print(f"Train loss: {train_loss:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Test loss: {test_loss:.4f}")
