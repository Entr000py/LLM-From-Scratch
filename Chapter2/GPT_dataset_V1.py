import torch
from torch.utils.data import Dataset
import tiktoken
from torch.utils.data import DataLoader


class GPTDataset(Dataset):
    """
    用于GPT模型训练的自定义数据集。
    它将长文本切分成固定长度的输入(input)和目标(target)序列。
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        """
        初始化数据集。

        参数:
            txt (str): 待处理的原始文本字符串。
            tokenizer: 用于将文本编码为token ID的分词器。
            max_length (int): 每个输入序列的最大长度（即上下文窗口大小）。
            stride (int): 滑动窗口的步长。
        """
        self.input_ids = []
        self.target_ids = []

        # 将整个文本编码为token ID列表
        token_ids = tokenizer.encode(txt)

        # 使用滑动窗口从token_ids中创建输入和目标块
        # 循环的起点是0，终点是确保最后一个块也能完整取出
        for i in range(0, len(token_ids) - max_length, stride):
            # 提取输入块 (例如: [0, 1, 2, ..., 1023])
            input_chunk = token_ids[i:i + max_length]
            # 提取目标块，它是输入块向右平移一位的结果 (例如: [1, 2, 3, ..., 1024])
            # 这样，对于每个输入的token，模型都需要预测下一个token
            target_chunk = token_ids[i + 1:i + max_length + 1]
            
            # 将块转换为PyTorch张量并添加到列表中
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """返回数据集中样本的总数。"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        根据索引idx获取一个样本。

        返回:
            一个元组 (input_ids, target_ids)，分别对应输入和目标序列。
        """
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(txt, tokenizer, max_length=256, stride=128, batch_size=4, shuffle=True, drop_last=True, num_workers=0):
    """
    创建一个数据加载器。

    参数:
        txt (str): 待处理的原始文本字符串。
        tokenizer: 用于将文本编码为token ID的分词器。
        max_length (int): 每个输入序列的最大长度。
        stride (int): 滑动窗口的步长。
        batch_size (int): 每个批次的样本数量。
        shuffle (bool): 是否在每个epoch中打乱数据集。
        drop_last (bool): 如果数据集大小不能被batch_size整除，是否丢弃最后一个不完整的批次。
        num_workers (int): 用于数据加载的子进程数量。默认为0，表示在主进程中加载。
    返回:
        PyTorch DataLoader对象。
    """
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

if __name__ == "__main__":
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    max_length =4

    dataloader = create_dataloader_v1(
        raw_text, tokenizer, batch_size=8, max_length=max_length, stride=max_length, shuffle=False, 
        )
    
    vocab_size = 50257
    output_dim = 256
    
    # 将dataloader转换为一个迭代器，这样我们就可以逐批次地从中提取数据
    data_iter = iter(dataloader)
    # 从迭代器中获取下一个项目，也就是第一个批次的数据
    first_batch = next(data_iter)

    print("--- First Batch ---")
    input_ids, target_ids = first_batch
    inputs, targets = next(data_iter)
    print("INPUTS:", inputs)
    print("INPUTS shape:", inputs.shape)
    print("TARGETS:", targets)

    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)    #查找运算
    embedding = torch.nn.Embedding(vocab_size, output_dim)

    token_embeddings = embedding_layer(inputs)
    print("TOKEN EMBEDDINGS:", token_embeddings)
    print("TOKEN EMBEDDINGS shape:", token_embeddings.shape) #torch.Size([8, 4, 256])

    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length)) #提供0到context_length-1的序列，输出对应位置的编码
    print("POS EMBEDDINGS shape:", pos_embeddings.shape) #位置编码

    input_embeddings = token_embeddings + pos_embeddings #位置编码和token编码相加
    print("INPUT EMBEDDINGS shape:", input_embeddings.shape)
    

    
