# LLM-From-Scratch

本项目旨在从零开始逐步构建一个大型语言模型（LLM），参考了 Andrej Karpathy 的 "Let's build GPT" 教程。项目代码包含了构建和训练一个类 GPT 模型所需的关键组件，例如分词、数据集构建和数据加载。

## 项目结构

```
LLM-From-Scratch/
├───requirements.in         # pip-compile 的输入文件
├───requirements.txt        # 项目依赖
├───Chapter2/               # 第二章：数据处理与分词
│   ├───byte_pair_encoding.py # 演示 BPE 分词器如何编码和解码文本
│   ├───GPT_dataset_V1.py   # 为 GPT 模型构建数据集和数据加载器
│   └───the-verdict.txt     # 用于训练和演示的示例文本
└───README.md               # 项目说明
```

## 主要内容

### Chapter2: 数据集与分词

本章节重点关注为模型训练准备数据。

- **`byte_pair_encoding.py`**:
  这个脚本展示了如何使用 `tiktoken` 库（由 OpenAI 开发）来进行字节对编码（Byte Pair Encoding, BPE）。BPE 是一种常见的分词算法，能够有效地将文本切分为模型可以理解的 token。

- **`GPT_dataset_V1.py`**:
  该文件定义了一��自定义的 PyTorch `Dataset` 类 (`GPTDataset`) 和一个数据加载器 (`create_dataloader_v1`)。它的核心功能是：
  1.  接收长文本作为输入。
  2.  使用分词器将文本编码为 token ID。
  3.  通过滑动窗口（sliding window）的方式，将长串的 token 切分为固定长度的 `(input, target)` 对。其中 `target` 是 `input` 向右平移一个位置的结果，这是语言模型进行“下一个词预测”任务的标准做法。

## 环境设置

1.  **克隆项目**
    ```bash
    git clone <your-repository-url>
    cd LLM-From-Scratch
    ```

2.  **安装依赖**
    项目依赖项已在 `requirements.txt` 中列出。建议创建一个虚拟环境，然后运行以下命令安装：
    ```bash
    pip install -r requirements.txt
    ```
    主要依赖包括：
    - `pytorch`: 用于构建和训练模型。
    - `tiktoken`: 用于文本分词。
    - `pandas` & `numpy`: 数据处理。

## 如何运行

你可以直接运行 `Chapter2` 中的脚本来查看其效果。

- **运行数据加载器示例**:
  此命令将使用 `the-verdict.txt` 文件作为原始数据，创建一个数据加载器，并打印出第一个批次（batch）的输入和目标数据，以展示数据是如何为模型准备的。
  ```bash
  python Chapter2/GPT_dataset_V1.py
  ```

- **运行 BPE 编码示例**:
  此命令将演示如何使用 `gpt2` 的分词器来编码和解码一段简单的文本。
  ```bash
  python Chapter2/byte_pair_encoding.py
  ```
