import torch
import torch.nn as nn
from attention_with_trainable_weights import SelfAttention_v2

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
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
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)

#dropout方法，归一化之后再处理
dropout = nn.Dropout(p=0.5)
print(dropout(attn_weights))

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)