import torch
import torch.nn as nn
import tiktoken

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

class DummyGPTmodel(nn.Module):
    def __init__(self, cfg):
        super().__init__() # 调用nn.Module的构造函数

        # Token 嵌入层：将输入整数索引（token ID）转换为连续的向量表示
        # 输入：(batch_size, seq_len) 的 token ID
        # 输出：(batch_size, seq_len, emb_dim) 的 token 嵌入向量
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # 位置嵌入层：为序列中的每个位置提供唯一的向量表示，
        # 允许模型理解token的顺序信息。
        # 输入：(seq_len) 的位置索引
        # 输出：(seq_len, emb_dim) 的位置嵌入向量
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # 嵌入Dropout层：在token嵌入和位置嵌入相加后应用Dropout
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Transformer 块序列：这是模型的核心，包含多个Transformer解码器块。
        # nn.Sequential 允许按顺序堆叠多个模块。
        # 这里使用列表推导式创建 cfg["n_layers"] 个 DummyTransformerBlock 实例。
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # 最终的Layer Normalization层：在Transformer块的输出之后应用，
        # 有助于稳定训练并提高性能。
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])

        # 输出头（线性层）：将模型的内部表示投影回词汇表大小，
        # 用于生成每个token的逻辑（logits），以便进行下一个token的预测。
        # bias=False 是GPT-2的常用设置。
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """
        GPT模型的前向传播过程。
        
        Args:
            in_idx (torch.Tensor): 输入的token索引张量，形状为 (batch_size, seq_len)。
        
        Returns:
            torch.Tensor: 模型的输出逻辑（logits），形状为 (batch_size, seq_len, vocab_size)。
                          这些logits可以用于计算下一个token的概率。
        """
        # 1. 获取输入张量的批次大小和序列长度
        batch_size, seq_len = in_idx.shape

        # 2. Token 嵌入
        # 将输入的token索引转换为对应的嵌入向量
        # tok_embeds 形状：(batch_size, seq_len, emb_dim)
        tok_embeds = self.tok_emb(in_idx)

        # 3. 位置嵌入
        # 生成一个从0到seq_len-1的序列，作为位置索引
        # pos_embeds 形状：(seq_len, emb_dim)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        # 4. 嵌入相加
        # 将token嵌入和位置嵌入相加。PyTorch会自动广播 pos_embeds 到 batch_size。
        # x 形状：(batch_size, seq_len, emb_dim)
        x = tok_embeds + pos_embeds

        # 5. 嵌入Dropout
        # 对融合后的嵌入应用Dropout
        # x 形状保持不变：(batch_size, seq_len, emb_dim)
        x = self.drop_emb(x)

        # 6. Transformer 块处理
        # 将数据依次通过所有 Transformer 解码器块。
        # 在这个占位模型中，DummyTransformerBlock 只是简单地返回输入，
        # 但在真实模型中，这里会进行自注意力、层归一化、前馈网络等复杂计算。
        # x 形状保持不变：(batch_size, seq_len, emb_dim)
        x = self.trf_blocks(x)

        # 7. 最终的Layer Normalization
        # 对Transformer块的输出应用Layer Normalization。
        # x 形状保持不变：(batch_size, seq_len, emb_dim)
        x = self.final_norm(x)

        # 打印 mean 和 var
        mean = x.mean()
        var = x.var()
        print("mean:", mean)
        print("var:", var)

        # 8. 输出头
        # 将处理后的内部表示通过线性层投影到词汇表大小的维度，
        # 得到每个位置上每个词元的预测逻辑。
        # logits 形状：(batch_size, seq_len, vocab_size)
        logits = self.out_head(x)

        return logits

class DummyTransformerBlock(nn.Module):
        def __init__(self, cfg):
            super().__init__()

        def forward(self, x):
            return x

class DummylayerNorm(nn.Module):
        def __init__(self, normalized_shape, eps=1e-5):
            super().__init__()

        def forward(self, x):
            return x

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm + self.shift


tokenizer = tiktoken.get_encoding("gpt2")
batch = []
text1 = "Every effort moves you"
text2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(text1)))
batch.append(torch.tensor(tokenizer.encode(text2)))
batch = torch.stack(batch)

if __name__ == "__main__":
    #print(batch)
    torch.manual_seed(123)
    model = DummyGPTmodel(GPT_CONFIG_124M)
    logits = model(batch)

    batch_example = torch.randn(2, 5)

    #normalization

    # layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    # out = layer(batch_example)
    # mean = out.mean(dim=-1, keepdim = True)
    # var = out.var(dim=-1, keepdim = True)
    # torch.set_printoptions(sci_mode=False)
    # out_norm = (out - mean) / torch.sqrt(var)
    # mean = out_norm.mean(dim=-1, keepdim = True)
    # var = out_norm.var(dim=-1, keepdim = True)
    # print(mean)
    # print(var)

    ln = LayerNorm(emb_dim= 5)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim = True)
    var = out_ln.var(dim=-1, unbaised = False, keepdim = True)
    print(mean)
    print(var)



