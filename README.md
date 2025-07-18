# LLM-From-Scratch

This project is a step-by-step guide to building a GPT-like Large Language Model (LLM) from scratch, following the structure of the book "Build a Large Language Model (From Scratch)" by Sebastian Raschka. Each chapter's code is organized into its own directory, demonstrating the core concepts of building an LLM.

## Project Structure

The project is divided into chapters, each focusing on a specific aspect of building an LLM:

- **Chapter 2: Data Preparation and Tokenization**
  - `byte_pair_encoding.py`: Implements the Byte-Pair Encoding (BPE) tokenizer.
  - `GPT_dataset_V1.py`: Creates a custom dataset for training the GPT model.
  - `token_embeddings.py`: Demonstrates how to create token and positional embeddings.

- **Chapter 3: Attention Mechanisms**
  - `self_attention.py`: Implements the basic self-attention mechanism.
  - `attention_with_trainable_weights.py`: Adds trainable weights to the self-attention mechanism.
  - `casual_attention.py`: Implements causal attention to prevent the model from looking at future tokens.
  - `multi_head_attention.py`: Implements multi-head attention to allow the model to focus on different parts of the input sequence.

- **Chapter 4: Building the GPT Architecture**
  - `GELU.py`: Implements the GELU activation function.
  - `transformer.py`: Implements the Transformer block, which is the core component of the GPT model.
  - `shortcut_connection.py`: Demonstrates the use of shortcut (residual) connections to improve gradient flow.
  - `GPT_architecture.py`: Puts together the complete GPT model architecture.
  - `GPTmodel.py`: A complete GPT model implementation.

- **Chapter 5: Training and Text Generation**
  - `train_model.py`: Contains the code for training the GPT model.
  - `generate_text.py`: Contains the code for generating text using the trained model.

- **dataset/**
  - `the-verdict.txt`: The text data used for training the model.

## Getting Started

### Prerequisites

- Python 3.8 or later
- PyTorch
- tiktoken
- matplotlib

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/LLM-From-Scratch.git
    cd LLM-From-Scratch
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Code

You can run the code for each chapter individually. For example, to run the code for training the model in Chapter 5, you can execute the following command:

```bash
python Chapter5/train_model.py
```

This will train the GPT model on the provided dataset and save the trained model to `model.pth`. It will also generate a plot of the training and validation loss in `training_loss_plot.png`.

To generate text using the trained model, you can run:

```bash
python Chapter5/generate_text.py
```

## Concepts Covered

This project provides a hands-on implementation of the following core concepts of LLMs:

- **Tokenization:** Byte-Pair Encoding (BPE)
- **Embeddings:** Token and Positional Embeddings
- **Attention Mechanisms:** Self-Attention, Causal Attention, Multi-Head Attention
- **Transformer Architecture:** Transformer Blocks, GELU Activation, Layer Normalization, Shortcut Connections
- **Training:** Custom Datasets, DataLoaders, Training Loop, Loss Calculation
- **Text Generation:** Greedy Decoding

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you find any bugs or have suggestions for improvement.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.