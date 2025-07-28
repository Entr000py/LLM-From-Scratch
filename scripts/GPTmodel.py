from transformer import TransformerBlock 
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
        self.out_head.weight = self.tok_emb.weight  #节省内存和计算资源

    def forward(self, in_idx):
        """
        GPT模型的前向传播。

        参数:
            in_idx (torch.Tensor): 输入token的索引，形状为 (batch_size, sequence_length)。

        返回:
            torch.Tensor: 模型的输出logits，形状为 (batch_size, sequence_length, vocab_size)。
        """
        batch_size, seq_len = in_idx.shape
        # 将输入token索引转换为token嵌入
        tok_embeds = self.tok_emb(in_idx)
        # 生成位置嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # 将token嵌入和位置嵌入相加
        x = tok_embeds + pos_embeds
        # 应用dropout
        x = self.drop_emb(x)
        # 通过Transformer块序列
        x = self.trf_blocks(x)
        # 应用最终的层归一化
        x = self.final_norm(x)
        # 通过输出层得到logits
        logits = self.out_head(x)
        return logits

def generate_text_simple(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None):
    """
    使用给定的GPT模型生成文本。

    参数:
        model (nn.Module): GPT模型实例。
        idx (torch.Tensor): 初始的输入token序列，形状为 (batch_size, sequence_length)。
        max_new_tokens (int): 要生成的最大新token数量。
        context_size (int): 模型用于预测下一个token的上下文大小。
        temperature (float, optional): 用于调整模型输出概率分布的温度。较高的温度使分布更平坦（更多随机性），较低的温度使分布更尖锐（更确定性）。默认为1.0。
        top_k (int, optional): 限制采样时只考虑概率最高的k个token。默认为None（不限制）。

    返回:
        torch.Tensor: 包含原始输入和生成的新token的完整序列。
    """
    # 循环生成指定数量的新token
    for _ in range(max_new_tokens):
        # 截取输入序列的最后context_size个token作为当前上下文
        idx_cond = idx[:, -context_size:]
        
        # 在不计算梯度的模式下进行推理，以节省内存和计算
        with torch.no_grad():
            # 使用模型预测下一个token的logits
            logits = model(idx_cond)
        
        # 获取最后一个时间步的logits，并应用温度缩放
        logits = logits[:, -1, :] / temperature

        # 如果指定了top_k，则只保留概率最高的k个token的logits，其余设置为负无穷
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # 将logits转换为概率分布
        probs = torch.softmax(logits, dim=-1)
        # 从概率分布中多项式采样下一个token的索引
        idx_next = torch.multinomial(probs, num_samples=1)
        # 将新生成的token添加到序列中
        idx = torch.cat((idx, idx_next), dim=1)

    # 返回包含生成文本的完整token序列
    return idx

if __name__ == "__main__":
    torch.manual_seed(123)
    model = GPTmodel(GPT_CONFIG_124M)

    batch = []
    text1 = "Every effort moves you"
    text2 = "Every day holds a"
    tokenizer = tiktoken.get_encoding("gpt2")
    batch.append(torch.tensor(tokenizer.encode(text1)))
    batch.append(torch.tensor(tokenizer.encode(text2)))
    batch = torch.stack(batch)

    out = model(batch)

    total_prams = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_prams}")   #共享权重，嵌入层的权重运用于输出层
    print("Token embeding shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)

    total_prams_2 = total_prams - sum(p.numel() for p in model.out_head.parameters())
    print(f"Total parameters without output layer: {total_prams_2}")

    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) #在索引为0的维度增加一个维度
    print("encoded_tensor:", encoded_tensor)

    model.eval()
    out = generate_text_simple(
        model = model,
        idx = encoded_tensor,
        max_new_tokens = 6,
        context_size = GPT_CONFIG_124M["context_length"]
    )
    print("Output:", out)
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print("Decoded text:", decoded_text)
