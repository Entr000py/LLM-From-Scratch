import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"])
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# if __name__ == "__main__":
#     from GPT_architecture import GPT_CONFIG_124M # 仅在需要时导入
#     ffn = FeedForward(GPT_CONFIG_124M)
#     x = torch.rand(2, 3, 768)
#     y = ffn(x)
#     print(y.shape)  #torch.Size([2, 3, 768])
