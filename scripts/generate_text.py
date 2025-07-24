import torch
import tiktoken
import sys
from pathlib import Path

# Add the project root to sys.path to allow importing modules from Chapter4
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import torch
import tiktoken
import sys
from pathlib import Path

# Add the project root to sys.path to allow importing modules
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from GPTmodel import GPTmodel, generate_text_simple

def text_to_ids(text, tokenizer):
    """Encodes text to token IDs."""
    return tokenizer.encode(text)

def ids_to_text(token_ids, tokenizer):
    """Decodes token IDs back to text."""
    ids = token_ids.squeeze()
    if hasattr(ids, 'tolist'):
        ids = ids.tolist()
    return tokenizer.decode(ids)

def batch_ids_to_text(token_ids_batch, tokenizer):
    """Decodes a batch of token IDs into a list of text strings."""
    if hasattr(token_ids_batch, 'tolist'):
        token_ids_batch = token_ids_batch.tolist()
    return [tokenizer.decode(ids) for ids in token_ids_batch]

if __name__ == "__main__":
    # This block is for demonstrating the text utility functions
    
    # 1. Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # 2. Demonstrate text_to_ids and ids_to_text
    text = "Hello, world!"
    token_ids = text_to_ids(text, tokenizer)
    print(f"Original text: '{text}'")
    print(f"Encoded token IDs: {token_ids}")
    decoded_text = ids_to_text(torch.tensor(token_ids), tokenizer)
    print(f"Decoded text: '{decoded_text}'")

    # 3. Demonstrate batch decoding
    batch_of_ids = [
        text_to_ids("This is the first sentence.", tokenizer),
        text_to_ids("This is the second one.", tokenizer)
    ]
    # To make it a tensor, we would need to pad sequences to the same length,
    # so we'll just use a list of lists for this demonstration.
    decoded_batch = batch_ids_to_text(batch_of_ids, tokenizer)
    print("\nBatch decoding demonstration:")
    for i, sentence in enumerate(decoded_batch):
        print(f"  Sentence {i+1}: '{sentence}'")

    # 4. Demonstrate simple text generation (requires a trained model)
    print("\nSimple text generation demonstration:")
    
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    model = GPTmodel(GPT_CONFIG_124M)
    model.eval()

    start_context = "Every effort moves you"
    encoded_input = torch.tensor(text_to_ids(start_context, tokenizer)).unsqueeze(0)
    
    token_ids = generate_text_simple(
        model=model,
        idx=encoded_input,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    
    print(f"Input context: '{start_context}'")
    print(f"Generated text: '{ids_to_text(token_ids, tokenizer)}'")

