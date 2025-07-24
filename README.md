# LLM-From-Scratch

This project provides a step-by-step guide to building a GPT-like Large Language Model (LLM) from scratch, inspired by the book "Build a Large Language Model (From Scratch)" by Sebastian Raschka. It demonstrates the core concepts of building and training an LLM using PyTorch.

## Features

- **Modular Codebase**: Core components like attention mechanisms, the Transformer architecture, and data loaders are organized into separate, easy-to-understand scripts.
- **Multi-GPU Training**: Supports training on multiple GPUs using `torch.nn.DataParallel` to accelerate the training process.
- **Logging**: Integrated logging for monitoring training progress and debugging.
- **Text Generation**: Includes scripts for generating new text with the trained model, featuring both simple and temperature-scaled sampling.
- **Data Handling**: Custom PyTorch `Dataset` and `DataLoader` for efficient text data processing.

## Project Structure

The project's scripts are located in the `scripts/` directory:

-   `GPTmodel.py`: The complete GPT model architecture.
-   `train_model.py`: The main script for training the model. It handles the training loop, evaluation, and saving results.
-   `generate_text.py`: Contains utility functions for text encoding/decoding and generation.
-   `GPT_dataset_V1.py`: Implements the custom PyTorch `Dataset` for the text data.
-   `utils.py`: Contains shared utility functions, such as loading text data.
-   `attention*.py`, `transformer.py`, etc.: Various scripts implementing the core building blocks of the Transformer model.

The training data is located in the `dataset/` directory.

## Getting Started

### Prerequisites

-   Python 3.8+
-   PyTorch
-   The packages listed in `requirements.txt`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/LLM-From-Scratch.git
    cd LLM-From-Scratch
    ```

2.  **Install the required packages:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

### How to Run

#### Training the Model

To train the model, run the `train_model.py` script:

```bash
python scripts/train_model.py
```

-   **Multi-GPU Usage**: The script will automatically detect and use all available GPUs. If you have 3 GPUs, it will use `DataParallel` to distribute the workload across them.
-   **Logging**: Training progress will be logged to the console.
-   **Output**: A plot of the training and validation loss will be saved as `training_loss_plot.png`.

#### Generating Text

The `generate_text.py` script can be run to see a demonstration of its utility functions. However, the primary text generation after training happens at the end of the `train_model.py` script.

## Recent Improvements

-   **Multi-GPU Support**: The training script was updated to automatically leverage multiple GPUs, specifically configured for up to 3 GPUs, to speed up training.
-   **Logging Implementation**: Replaced `print` statements with a robust logging system to provide structured and informative output during training.
-   **Code Refactoring**:
    -   Created a `scripts/utils.py` file for shared utility functions to reduce code duplication.
    -   Moved loss calculation logic from `generate_text.py` to `train_model.py` to improve separation of concerns.
    -   Cleaned up and clarified the responsibilities of each script.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
