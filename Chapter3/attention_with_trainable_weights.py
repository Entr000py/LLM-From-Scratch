import torch
import torch.nn as nn

class self_attention_v1(nn.Module):
  def __init__(self, d_in, d_out):
    super().__init__()
    self.d_out = d_out
    self.w_query = nn.Parameter(torch.randn(d_in, d_out))
    self.w_key = nn.Parameter(torch.randn(d_in, d_out))
    self.w_value = nn.Parameter(torch.randn(d_in, d_out))

  def forward(self, x):
    keys = x @ self.w_key
    queries = x @ self.w_query
    values = x @ self.w_value
    attn_scores = queries @ keys.T
    atten_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    context_vec = atten_weights @ values
    return context_vec

class SelfAttention_v2(nn.Module):
  def __init__(self, d_in, d_out):
    super().__init__()
    self.d_out = d_out
    self.w_query = nn.Parameter(torch.randn(d_in, d_out))
    self.w_key = nn.Parameter(torch.randn(d_in, d_out))
    self.w_value = nn.Parameter(torch.randn(d_in, d_out))

  def forward(self, x):
    keys = x @ self.w_key
    queries = x @ self.w_query
    values = x @ self.w_value
    attn_scores = queries @ keys.T
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    context_vec = attn_weights @ values
    return context_vec


# 定义输入张量inputs，模拟6个词的嵌入向量，每个词由3个特征表示
# 每个行向量代表一个词的嵌入，例如 "Your", "journey", "starts", "with", "one", "step"
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

print(inputs) # 打印输入张量，查看其内容

x_2 = inputs[1] # 获取输入张量中的第二个词向量 (journey)
d_in = inputs.shape[1] # 获取输入特征的维度，即每个词向量的特征数量 (3)
d_out = 2 # 定义输出特征的维度，即查询、键、值向量的维度 (2)

# torch.manual_seed(123) # 设置PyTorch的随机种子，以确保结果的可复现性

# # 定义可训练的权重矩阵，用于将输入词向量转换为查询、键、值向量
# # requires_grad=False 表示这些权重在训练过程中不会被优化（这里仅用于演示）
# w_query = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False) # 查询权重矩阵
# w_key = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
# w_value = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False) 

# # 计算第二个词向量 (x_2) 对应的查询、键、值向量
# query_2 = x_2 @ w_query # x_2 与查询权重矩阵相乘，得到查询向量
# key_2 = x_2 @ w_key     
# value_2 = x_2 @ w_value 

# # print(query_2) 


# keys = inputs @ w_key   # 所有输入词向量与键权重矩阵相乘，得到所有键向量
# values = inputs @ w_value 

# print("values.shape", values.shape) 
# print("keys.shape", keys.shape)    

# #注意力得分
# keys_2 = keys[1]
# attn_score_2 = query_2.dot(keys_2)
# print("attn_score_22", attn_score_2)

# d_k = keys.shape[-1]
# atten_weights_2 = torch.softmax(attn_score_2 / d_k**0.5, dim=-1)
# print(atten_weights_2)

# context_vec_2 = attn_weights_2 @ values
# print(context_vec_2)

#举例
torch.manual_seed(123)
sa_v1 = self_attention_v1(d_in, d_out)
print(sa_v1(inputs))
