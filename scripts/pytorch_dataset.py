import pandas as pd
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
import time

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
    model_to_load = gpt.module if isinstance(gpt, nn.DataParallel) else gpt
    model_to_load.pos_emb.weight = assign(model_to_load.pos_emb.weight, params['wpe'])               #A
    model_to_load.tok_emb.weight = assign(model_to_load.tok_emb.weight, params['wte'])
    for b in range(len(params["blocks"])):                                       #B
        q_w, k_w, v_w = np.split(                                                #C
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        model_to_load.trf_blocks[b].att.W_query.weight = assign(
            model_to_load.trf_blocks[b].att.W_query.weight, q_w.T)
        model_to_load.trf_blocks[b].att.W_key.weight = assign(
            model_to_load.trf_blocks[b].att.W_key.weight, k_w.T)
        model_to_load.trf_blocks[b].att.W_value.weight = assign(
            model_to_load.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        model_to_load.trf_blocks[b].att.W_query.bias = assign(
            model_to_load.trf_blocks[b].att.W_query.bias, q_b)
        model_to_load.trf_blocks[b].att.W_key.bias = assign(
            model_to_load.trf_blocks[b].att.W_key.bias, k_b)
        model_to_load.trf_blocks[b].att.W_value.bias = assign(
            model_to_load.trf_blocks[b].att.W_value.bias, v_b)

        model_to_load.trf_blocks[b].att.out_proj.weight = assign(
            model_to_load.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        model_to_load.trf_blocks[b].att.out_proj.bias = assign(
            model_to_load.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        model_to_load.trf_blocks[b].ff.layers[0].weight = assign(
            model_to_load.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        model_to_load.trf_blocks[b].ff.layers[0].bias = assign(
            model_to_load.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        model_to_load.trf_blocks[b].ff.layers[2].weight = assign(
            model_to_load.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        model_to_load.trf_blocks[b].ff.layers[2].bias = assign(
            model_to_load.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        model_to_load.trf_blocks[b].norm1.weight = assign(
            model_to_load.trf_blocks[b].norm1.weight,
            params["blocks"][b]["ln_1"]["g"])
        model_to_load.trf_blocks[b].norm1.bias = assign(
            model_to_load.trf_blocks[b].norm1.bias,
            params["blocks"][b]["ln_1"]["b"])
        model_to_load.trf_blocks[b].norm2.weight = assign(
            model_to_load.trf_blocks[b].norm2.weight,
            params["blocks"][b]["ln_2"]["g"])
        model_to_load.trf_blocks[b].norm2.bias = assign(
            model_to_load.trf_blocks[b].norm2.bias,
            params["blocks"][b]["ln_2"]["b"])
    
    model_to_load.final_norm.weight = assign(model_to_load.final_norm.weight, params["g"])
    model_to_load.final_norm.bias = assign(model_to_load.final_norm.bias, params["b"])

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

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # 将模型设置为评估模式，禁用 Dropout 和 BatchNorm 等层
    with torch.no_grad():  # 在此上下文管理器中禁用梯度计算，以节省内存并加快推理
        # 计算训练集上的损失
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        
        # 检查验证集是否为空，如果为空则验证损失为0
        if len(val_loader) == 0:
            val_loss = 0.0
        else:
            # 计算验证集上的损失
            val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # 将模型重新设置为训练模式
    return train_loss, val_loss  # 返回训练损失和验证损失

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

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, tokenizer):
    """
    训练一个简单的分类器模型。

    Args:
        model (nn.Module): 要训练的 PyTorch 模型。
        train_loader (DataLoader): 训练数据加载器。
        val_loader (DataLoader): 验证数据加载器。
        optimizer (Optimizer): 优化器。
        device (torch.device): 模型和数据所在的设备（例如 'cpu' 或 'cuda'）。
        num_epochs (int): 训练的总 epoch 数。
        eval_freq (int): 每隔多少个全局步长评估一次模型。
        eval_iter (int): 评估时使用的批次数量。
        tokenizer: 文本分词器（在此函数中未使用，但作为参数传递）。

    Returns:
        tuple: 包含训练损失、验证损失、训练准确率、验证准确率列表以及已处理的样本总数。
    """
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    example_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()  # 将模型设置为训练模式
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 清除之前的梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算当前批次的损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数
            example_seen += input_batch.shape[0]  # 累加已处理的样本数
            global_step += 1  # 增加全局步长

            if global_step % eval_freq == 0:  # 定期评估模型
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d})):"
                    f"Train loss {train_loss:.4f}, Val loss {val_loss:.4f}")

        # 每个 epoch 结束时计算训练和验证准确率
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% |", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, example_seen

def classify_review(text, model, tokenizer, device, max_length = None, pad_token_id = 50256):
    """
    对给定的文本评论进行分类（垃圾邮件或非垃圾邮件）。

    Args:
        text (str): 要分类的文本评论。
        model (nn.Module): 用于分类的 PyTorch 模型。
        tokenizer: 用于编码文本的 tokenizer 对象。
        device (torch.device): 模型和数据所在的设备（例如 'cpu' 或 'cuda'）。
        max_length (int, optional): 编码文本的最大长度。如果为 None，则使用模型支持的最大上下文长度。
        pad_token_id (int, optional): 用于填充短序列的 token ID。默认为 50256。

    Returns:
        str: 分类结果，"spam" 表示垃圾邮件，"not spam" 表示非垃圾邮件。
    """
    model.eval()  # 将模型设置为评估模式

    input_ids = tokenizer.encode(text)  # 使用 tokenizer 对文本进行编码
    # 处理 DataParallel 模型，获取实际的模型实例
    model_to_use = model.module if isinstance(model, nn.DataParallel) else model
    # 获取模型支持的上下文长度
    supported_context_length = model_to_use.pos_emb.weight.shape[1]

    # 截断或填充输入 ID 到指定或支持的最大长度
    input_ids = input_ids[:min(max_length, supported_context_length) if max_length is not None else supported_context_length]
    input_ids += [pad_token_id] * (max_length - len(input_ids)) if max_length is not None else []
    input_tensor = torch.tensor(input_ids, device = device).unsqueeze(0)  # 将输入 ID 转换为 Tensor 并添加批次维度

    with torch.no_grad():  # 在此块中禁用梯度计算，以节省内存和加快计算
        logits = model(input_tensor)[:, -1, :]  # 通过模型获取 logits，并选择最后一个时间步的输出
    predicted_label = torch.argmax(logits, dim = -1).item()  # 获取预测标签

    return "spam" if predicted_label == 1 else "not spam"  # 根据预测标签返回分类结果

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
        shuffle = False, # 验证集不打乱
        num_workers = num_workers,
        drop_last = True
    )
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False, # 测试集不打乱
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
        model = nn.DataParallel(model)

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

    for param in model.module.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.out_head.parameters():
        param.requires_grad = True

    # 获取自定义 GPT 模型所需的参数
    settings, params = download_and_load_gpt2(model_size="124M", models_dir=download_path)
    print("Available keys in params:", params.keys())
    
    # 将下载的权重加载到模型中
    load_weights_into_gpt(model.module, params)

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

    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 5
    train_losses, val_losses, train_accs, val_accs, example_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs = num_epochs, eval_freq = 50, eval_iter = 5, tokenizer = tokenizer
    )

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training completed in {execution_time:.2f} seconds.")

    text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
    )
    print(classify_review(
        text_1, model, tokenizer, device, max_length=train_dataset.max_length
    ))

    #保存模型
    torch.save(model.state_dict(), "review_classifier.pth")
