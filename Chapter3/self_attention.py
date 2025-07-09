import torch # 导入PyTorch库，用于张量操作

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

# 选择inputs中的第二个词向量（索引为1，即"journey"的嵌入）作为查询向量query
query = inputs[1]   #就是Q

# 计算查询向量query与所有输入向量的点积，得到注意力分数
# inputs的形状是(6, 3)，query的形状是(3,)
# inputs @ query 会得到形状为(6,)的张量，其中每个元素是inputs的行向量与query的点积
attn_scores_2 = inputs @ query
print(attn_scores_2.shape) #torch.Size([6])
print(attn_scores_2) #tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])

# 使用torch的softmax函数对注意力分数进行归一化
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# 计算上下文向量
# attn_weights_2的形状是(6,)，inputs的形状是(6, 3)
# attn_weights_2.unsqueeze(1) 将形状变为(6, 1)，然后进行广播乘法
# 最终对第一个维度求和，得到形状为(3,)的上下文向量
context_vec_2 = torch.sum(attn_weights_2.unsqueeze(1) * inputs, dim=0)
print("Context vector:", context_vec_2)

# 计算所有输入向量之间的注意力分数矩阵 (Query-Key点积)
# inputs的形状是(6, 3)，inputs.T的形状是(3, 6)
# inputs @ inputs.T 会得到形状为(6, 6)的注意力分数矩阵
attn_scores = inputs @ inputs.T
print(attn_scores)

# 对注意力分数矩阵的最后一个维度进行softmax归一化，得到注意力权重矩阵
atten_weights = torch.softmax(attn_scores, dim=-1) #对最后一个维度进行归一化
print(atten_weights)

# 使用注意力权重矩阵和输入向量计算所有上下文向量
# atten_weights的形状是(6, 6)，inputs的形状是(6, 3)
# atten_weights @ inputs 会得到形状为(6, 3)的上下文向量矩阵
all_contex_vecs = atten_weights @ inputs
print(all_contex_vecs)
