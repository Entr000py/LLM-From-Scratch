import torch
import torch.nn as nn
from casual_attention import CasualAttention

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
    d_out =2
    torch.manual_seed(123)
    contex_length = batch.shape[1]
    mha = MultiHeadAttentionWrapper(d_in, d_out, contex_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print("contex_vecs.shape:",context_vecs.shape)  #contex_vecs.shape: torch.Size([2, 6, 4])

    a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],             
                        [0.8993, 0.0390, 0.9268, 0.7388],
                        [0.7179, 0.7058, 0.9156, 0.4340]],
                    [[0.0772, 0.3565, 0.1479, 0.5331],
                        [0.4066, 0.2318, 0.4545, 0.9737],
                        [0.4606, 0.5159, 0.4220, 0.5786]]]])

    first_head = a[0, 0, :, :]
    first_res = first_head @ first_head.T
    print("first_res:", first_res)
    second_head = a[0, 1, :, :]
    second_res = second_head @ second_head.T
    print("second_res:", second_res)

    # GPT-2 最小模型的参数
    gpt2_min_d_model = 768  # 输入/输出嵌入维度
    gpt2_min_num_heads = 12  # 注意力头数量
    gpt2_min_context_length = 1024 # 上下文长度
    gpt2_min_dropout_rate = 0.1 # Dropout 比率
    gpt2_qkv_bias = True # GPT-2 中 QKV 线性层带有偏置

    # 初始化多头注意力模块
    gpt2_min_attention = MultiHeadAttention(
        d_in=gpt2_min_d_model,
        d_out=gpt2_min_d_model, # 输出维度与输入维度相同
        contex_length=gpt2_min_context_length,
        dropout=gpt2_min_dropout_rate,
        num_heads=gpt2_min_num_heads,
        qkv_bias=gpt2_qkv_bias
    )
