import sys
import os

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from GELU import FeedForward
from Chapter3.multi_head_attention import MultiHeadAttention
from GPT_architecture import GPT_CONFIG_124M

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

if __name__ == "__main__":
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
