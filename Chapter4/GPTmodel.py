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

def generate_text_simple(model, idx, max_new_tokens, contex_size):  #贪心解码
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -contex_size:]    #取最后contex_size个token
        with torch.no_grad():   #只做推理
            logits = model(idx_cond)    #参数输入模型
        logits = logits[:, -1, :]   #取出最后一个位置的
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)    #返回概率最高的索引
        idx = torch.cat((idx, idx_next), dim=1) #新生成的token拼接到现有序列尾部，作为下次循环的上下文

    return idx


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
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor:", encoded_tensor)

model.eval()
out = generate_text_simple(
    model = model,
    idx = encoded_tensor,
    max_new_tokens = 6,
    contex_size = GPT_CONFIG_124M["context_length"]
)
print("Output:", out)
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print("Decoded text:", decoded_text)

