import torch
import torch.nn as nn
from attention_with_trainable_weights import SelfAttention_v2


class CasualAttention(nn.Module):
  mask: torch.Tensor
  def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
    super().__init__()
    self.d_out = d_out
    self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer(
      "mask",
      torch.triu(torch.ones(context_length, context_length), diagonal=1)
    )
  
  def forward(self, x):
    b, num_tokens, d_in = x.shape
    keys = self.w_key(x)
    queries = self.w_query(x) #queries 的形状是 (b, num_tokens, d_out)
    values = self.w_value(x)
    attn_scores = queries @ keys.transpose(1, 2)  #交换后两个参数的位置，计算注意力分数
    attn_scores.masked_fill_(self.mask.to(torch.bool)[:num_tokens, :num_tokens], -torch.inf)  #应用因果掩码
    attn_weights = torch.softmax(attn_scores / self.d_out**0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)
    context_vec = attn_weights @ values
    return context_vec



if __name__ == "__main__":
  inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your    (x^1)
  [0.55, 0.87, 0.66], # journey  (x^2)
  [0.57, 0.85, 0.64], # starts   (x^3)
  [0.22, 0.58, 0.33], # with     (x^4)
  [0.77, 0.25, 0.10], # one      (x^5)
  [0.05, 0.80, 0.55]] # step     (x^6)
)
  
  batch = torch.stack((inputs, inputs), dim=0)
  d_in = inputs.shape[1]
  d_out = 2

  sa_v2 = SelfAttention_v2(d_in, d_out)
  queries = sa_v2.w_query(inputs)
  keys = sa_v2.w_key(inputs)
  attn_scores = queries @ keys.T
  #print(attn_weights)

  # #传统掩码注意力矩阵
  # #下三角矩阵的掩码
  contex_length = attn_scores.shape[0]
  # mask_simple = torch.tril(torch.ones(contex_length, contex_length))
  # print(mask_simple)

  # #两个矩阵相乘
  # mask_simple = attn_weights * mask_simple
  # row_sums = mask_simple.sum(dim=[1], keepdim=True)
  # mask_simple_norm = mask_simple / row_sums
  # print(mask_simple_norm)

  #采用负无穷方法
  mask = torch.triu(torch.ones(contex_length, contex_length), diagonal=1)
  masked = attn_scores.masked_fill(mask.to(torch.bool), -torch.inf)
  print(masked)

  attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)

  #dropout方法，归一化之后再处理
  dropout = nn.Dropout(p=0.5)
  print(dropout(attn_weights))


  print(batch.shape)

  torch.manual_seed(123)
  contex_length = batch.shape[1]
  ca = CasualAttention(d_in, d_out, contex_length, 0.0)
  context_vecs = ca(batch)
  print("contex_vecs.shape:",context_vecs.shape)  #contex_vecs.shape: torch.Size([2, 6, 2])
