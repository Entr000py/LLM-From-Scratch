import torch
from GPT_dataset_V1 import create_dataloader_v1

input_ids = torch.tensor([[1, 2, 3, 4, 5]])

vocab_size = 50257
output_dim = 256

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)    #类似于查找运算

embedding = torch.nn.Embedding(vocab_size, output_dim)

print(embedding(input_ids))
print(embedding(input_ids).shape)
print(embedding_layer(torch.tensor([3])))

