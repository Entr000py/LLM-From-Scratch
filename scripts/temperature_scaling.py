"""
温度缩放演示脚本

该脚本演示了温度缩放如何影响语言模型中词汇的概率分布。
通过可视化不同温度值下的概率分布，展示温度对文本生成多样性的影响。
"""

import torch
from torch.cuda import temperature
import matplotlib.pyplot as plt


def print_sample_tokens(probas):
    """
    通过多项式采样生成并打印1000个样本token的统计结果
    
    :param probas: 词汇概率分布张量
    :return: None
    """
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples= 1).item() for i in range(1000)]
    sample_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sample_ids):
        print(f"{freq} * {inverse_vocab[i]}")

def softmax_with_temperature(logits, temperature):
    """
    使用指定温度值对logits进行缩放并应用softmax函数
    
    温度缩放原理：
    - 温度值较高时，概率分布更均匀，生成结果更多样化
    - 温度值较低时，概率分布更尖锐，生成结果更确定性
    
    :param logits: 原始logits值（未归一化的对数概率）
    :param temperature: 温度值，必须为正数
    :return: 经过温度缩放和softmax后的概率分布
    """
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

def generate(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eos_id=None):
    """
    使用给定模型生成文本序列
    
    Args:
        model: 用于生成文本的模型
        idx: 输入的token序列，形状为(batch_size, seq_len)
        max_new_tokens: 最大生成token数
        context_size: 上下文大小，用于限制输入序列长度
        temperature: 温度参数，控制生成的随机性
                    - temperature > 1: 更随机的输出
                    - temperature = 1: 原始分布
                    - temperature < 1: 更确定性的输出
        top_k: Top-K采样参数，只保留概率最高的K个token
        eos_id: 结束符ID，当生成此token时停止生成
        
    Returns:
        生成的token序列，包含原始输入和新生成的token
    """
    # 循环生成新的token，直到达到最大生成数量
    for _ in range(max_new_tokens):
        # 限制上下文长度，只取最后context_size个token
        idx_cond = idx[:, -context_size:]
        
        # 模型前向传播，获取logits
        with torch.no_grad():
            logits = model(idx_cond)
            
        # 只取最后一个位置的logits
        logits = logits[:, -1, :]
        
        # 如果指定了top_k，则进行top-k过滤
        if top_k is not None:
            # 获取top-k个最大的logits值
            top_logits, _ = torch.topk(logits, top_k)
            # 将小于第k大值的logits设为负无穷，实现top-k过滤
            logits = torch.where(
                logits < top_logits[..., -1, None],
                torch.tensor(float("-inf")).to(logits.device),
                logits
            )
        
        # 根据温度参数调整logits分布
        if temperature > 0.0:
            # 温度缩放：除以温度值
            logits = logits / temperature
            # 应用softmax获取概率分布
            probs = torch.softmax(logits, dim=-1)
            # 从概率分布中采样下一个token
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # 贪心解码：选择概率最高的token
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
        # 如果指定了结束符且生成了结束符，则停止生成
        if eos_id is not None and idx_next.item() == eos_id:
            break
            
        # 将新生成的token添加到序列中
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx

if __name__ == "__main__":
    # 定义词汇表，将词汇映射到索引
    vocab = {
        "closer": 0,
        "every": 1,
        "effort": 2,
        "forward": 3,
        "inches": 4,
        "moves": 5,
        "pizza": 6,
        "toward": 7,
        "you": 8,
    }
    # 创建反向词汇表，将索引映射到词汇
    inverse_vocab = {v: k for k, v in vocab.items()}

    # 定义下一个token的logits值（示例数据）
    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )

    # 设置随机种子以确保结果可重现
    torch.manual_seed(123)
    # 对logits应用softmax得到概率分布
    probas = torch.softmax(next_token_logits, dim=0)
    # 从概率分布中采样一个token
    next_token_id = int(torch.multinomial(probas, num_samples= 1).item())
    #print(inverse_vocab[next_token_id])
    
    # 打印1000次采样的统计结果
    print_sample_tokens(probas)
    
    # 定义不同的温度值用于比较
    temperature = [1, 0.1, 5]
    # 计算不同温度下的概率分布
    scaled_probs = [softmax_with_temperature(next_token_logits, t) for t in temperature]
    
    # 准备可视化数据
    x = torch.arange(len(vocab))
    bar_width = 0.15
    
    # 创建图表
    fig, ax = plt.subplots(figsize = (5, 3))
    for i, T in enumerate(temperature):
        rects = ax.bar(x + i * bar_width, scaled_probs[i],
        bar_width, label = f'Temperature = {T}')
    
    # 设置图表属性
    ax.set_ylabel('Probability')
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()
    
    #top-k
    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)
    new_logits = torch.where(
        condition = next_token_logits < top_logits[-1],
        input = torch.tensor(float("-inf")),
        other = next_token_logits
    )
    topk_probs = torch.softmax(new_logits, dim=0)
    print(topk_probs)

    # 调整布局并显示图表
    plt.tight_layout()
    plt.show()
