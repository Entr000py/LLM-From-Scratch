import torch
from torch.cuda import temperature
import matplotlib.pyplot as plt


def print_sample_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples= 1).item() for i in range(1000)]
    sample_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sample_ids):
        print(f"{freq} * {inverse_vocab[i]}")

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

if __name__ == "__main__":
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
    inverse_vocab = {v: k for k, v in vocab.items()}

    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )

    torch.manual_seed(123)
    probas = torch.softmax(next_token_logits, dim=0)
    next_token_id = int(torch.multinomial(probas, num_samples= 1).item())
    #print(inverse_vocab[next_token_id])
    print_sample_tokens(probas)
    temperature = [1, 0.1, 5]
    scaled_probs = [softmax_with_temperature(next_token_logits, t) for t in temperature]
    x = torch.arange(len(vocab))
    bar_width = 0.15
    fig, ax = plt.subplots(figsize = (5, 3))
    for i, T in enumerate(temperature):
        rects = ax.bar(x + i * bar_width, scaled_probs[i],
        bar_width, label = f'Temperature = {T}')
    ax.set_ylabel('Probability')
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.show()



