import pandas as pd
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gpt_download import download_and_load_gpt2
from GELU import FeedForward


class GPTmodel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #多头注意力机制
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out= cfg["emb_dim"],
            contex_length= cfg["context_length"],
            num_heads= cfg["n_heads"],
            dropout= cfg["drop_rate"],
            qkv_bias= cfg["qkv_bias"]
        )
        #前馈神经网络
        self.ff = FeedForward(cfg)
        #层归一化
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        #dropout
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, mask=None):
        #多头注意力
        shortcut = x    #保存原始输入用于快捷连接
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        #前馈神经网络
        shortcut = x    #保存第一个子层输出用于快捷连接
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

def load_weights_into_gpt(model, params):
    model.tok_emb.weight.data = params["wte"]
    model.pos_emb.weight.data = params["wpe"]
    model.final_norm.weight.data = params["ln_f.g"]
    model.final_norm.bias.data = params["ln_f.b"]

    for i, block in enumerate(model.trf_blocks):
        block.norm1.weight.data = params[f"h.{i}.ln_1.g"]
        block.norm1.bias.data = params[f"h.{i}.ln_1.b"]
        q_w, k_w, v_w = torch.chunk(params[f"h.{i}.attn.c_attn.w"], 3, dim=-1)
        q_b, k_b, v_b = torch.chunk(params[f"h.{i}.attn.c_attn.b"], 3, dim=-1)
        block.att.w_query.weight.data = q_w
        block.att.w_key.weight.data = k_w
        block.att.w_value.weight.data = v_w
        block.att.w_query.bias.data = q_b
        block.att.w_key.bias.data = k_b
        block.att.w_value.bias.data = v_b
        block.att.out_proj.weight.data = params[f"h.{i}.attn.c_proj.w"]
        block.att.out_proj.bias.data = params[f"h.{i}.attn.c_proj.b"]
        block.norm2.weight.data = params[f"h.{i}.ln_2.g"]
        block.norm2.bias.data = params[f"h.{i}.ln_2.b"]
        block.ff.c_fc.weight.data = params[f"h.{i}.mlp.c_fc.w"]
        block.ff.c_fc.bias.data = params[f"h.{i}.mlp.c_fc.b"]
        block.ff.c_proj.weight.data = params[f"h.{i}.mlp.c_proj.w"]
        block.ff.c_proj.bias.data = params[f"h.{i}.mlp.c_proj.b"]

class MultiHeadAttentionWrapper(torch.nn.Module):
    def __init__(self, d_in, d_out, contex_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CasualAttention(d_in, d_out, contex_length, dropout, qkv_bias) for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class MultiHeadAttention(nn.Module):
    # 类型提示：声明mask是一个torch.Tensor，这有助于代码的可读性和静态分析工具
    mask: torch.Tensor 

    def __init__(self, d_in, d_out, contex_length, dropout, num_heads, qkv_bias=False):
        """
        初始化多头注意力模块。
        
        Args:
            d_in (int): 输入特征的维度 (例如，词嵌入维度)。
            d_out (int): 输出特征的维度 (通常与d_in相同，或为d_model)。
            contex_length (int): 输入序列的最大长度，用于创建因果掩码。
            dropout (float): Dropout 比率，用于注意力权重。
            num_heads (int): 注意力头的数量。
            qkv_bias (bool, optional): 是否在Q, K, V的线性变换中添加偏置项。默认为False。
        """
        super().__init__() # 调用父类nn.Module的构造函数

        # 确保输出维度d_out可以被注意力头的数量num_heads整除
        assert d_out % num_heads == 0 

        self.d_out = d_out # 模块的输出维度
        self.heads = num_heads # 注意力头的数量
        # 计算每个注意力头的维度。总输出维度 d_out 被平均分配给每个头。
        self.head_dim = d_out // num_heads 

        # 定义用于计算 Query (Q), Key (K), Value (V) 的线性变换层
        # 这些层将输入 d_in 投影到 d_out (num_heads * head_dim) 维度
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 定义最终的输出投影层。在所有注意力头的输出拼接后，
        # 会通过这个层将维度从 d_out 再次投影回 d_out (如果 d_out == d_model的话)。
        # 这是Transformer标准多头注意力中的Wo层。
        self.out_proj = nn.Linear(d_out, d_out) 

        # Dropout 层，用于在注意力权重上进行正则化
        self.dropout = nn.Dropout(dropout)

        # 注册一个非训练参数 'mask' (注意力掩码)
        # torch.triu(..., diagonal=1) 创建一个上三角矩阵，主对角线以上为1，其余为0。
        # 这个掩码用于实现因果（或自回归）注意力，确保每个位置只能关注到当前及之前的位置。
        self.register_buffer(
            "mask", 
            torch.triu(torch.ones(contex_length, contex_length), diagonal=1)
        )

    def forward(self, x):
        """
        前向传播函数。
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, d_in)。
        
        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, sequence_length, d_out)。
        """
        # 获取输入张量的批次大小(b)、序列长度(num_tokens)和输入特征维度(d_in)
        b, num_tokens, d_in = x.shape 

        # 1. 计算 Query, Key, Value
        # 将输入x分别通过Q, K, V的线性层，得到形状为(b, num_tokens, d_out)的张量
        keys = self.w_key(x) 
        queries = self.w_query(x)
        values = self.w_value(x)

        # 2. 将Q, K, V分割成多头
        # 将d_out维度拆分为 (num_heads, head_dim)，并调整形状
        # 结果形状变为 (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.heads, self.head_dim)
        values = values.view(b, num_tokens, self.heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.heads, self.head_dim)

        # 3. 调整维度顺序，为批次化的矩阵乘法做准备
        # 将heads维度移到第二个位置，以便每个头可以独立进行批次化计算
        # 结果形状变为 (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 4. 计算注意力分数 (Queries @ Keys^T)
        # 对于每个头，计算Query和Key的点积。keys.transpose(2, 3)交换了num_tokens和head_dim
        # 结果形状变为 (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3) 

        # 5. 应用因果掩码 (Masking)
        # 获取与当前序列长度匹配的布尔类型掩码
        mask_bool = self.mask.to(torch.bool)[:num_tokens, :num_tokens]
        # 将掩码中为True（即未来位置）的分数设置为负无穷，
        # 这样在softmax后这些位置的权重将变为0，实现因果性。
        attn_scores.masked_fill_(mask_bool, -torch.inf) 

        # 6. 计算注意力权重 (Softmax)
        # 对注意力分数进行缩放（除以head_dim的平方根）以防止梯度消失，
        # 然后在最后一个维度（num_tokens，即关注哪个token）上应用softmax，
        # 将分数转换为概率分布。
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1) 
        # 应用Dropout到注意力权重
        attn_weights = self.dropout(attn_weights)

        # 7. 计算上下文向量 (Attention Weights @ Values)
        # 将注意力权重与Values进行矩阵乘法，得到每个头的上下文向量
        # 结果形状为 (b, num_heads, num_tokens, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # 再次交换维度，将heads维度移回第三个位置
        # 结果形状变为 (b, num_tokens, num_heads, head_dim)

        # 8. 拼接多头输出并进行最终投影
        # contiguous() 确保张量在内存中是连续的，view操作需要此条件
        # view() 将 (num_heads, head_dim) 维度展平回 d_out 维度
        # 结果形状变为 (b, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        
        # 通过输出投影层，将拼接后的结果再次线性变换，通常是为了维度对齐或进一步融合信息。
        context_vec = self.out_proj(context_vec) 

        return context_vec # 返回最终的上下文向量

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

        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data['text']
        ]

        if max_length is None:
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
            torch.tensor(encoded_text, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
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

if __name__ == '__main__':
    tokenizer = tiktoken.get_encoding('gpt2')
    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)
    train_dataset = SpamDataset(
        csv_files='../dataset/train.csv',
        max_length = None,
        tokenizer = tokenizer
    )
    val_dataset = SpamDataset(
        csv_files='../dataset/validation.csv',
        max_length = None,
        tokenizer = tokenizer
    )
    test_dataset = SpamDataset(
        csv_files='../dataset/test.csv',
        max_length = None,
        tokenizer = tokenizer
    )
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        drop_last = True
    )
    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        drop_last = True
    )
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        drop_last = True
    )
    for input_batch, target_batch in train_loader:
        pass
    
    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves"
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
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
        f"Dataset length {train_dataset.max_length} exceeds model's context "
        f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
        f"`max_length={BASE_CONFIG['context_length']}`"
    )

    download_path = r"D:\Program Files (x86)\weight" # 注意 r"" 表示原始字符串，避免反斜杠的转义问题
    model_name = "gpt2" # 或者 "gpt2-large", "gpt2-medium", "gpt2-xl"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=download_path)

    # 加载模型（这将自动下载权重到指定路径）
    model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=download_path)
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    model = GPTmodel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
