<div align="center">
  <h1 align="center">LLM-From-Scratch</h1>
  <p align="center">
    一个从零开始构建类 GPT 大语言模型 (LLM) 的项目
  </p>
</div>

<p align="center">
  <a href="https://github.com/13106/LLM-From-Scratch/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="项目许可">
  </a>
  <a href="https://github.com/13106/LLM-From-Scratch/blob/main/CONTRIBUTING.md">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="欢迎PR">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python版本">
  </a>
</p>

---

本项目提供了一个分步指南，用于从零开始构建一个类似 GPT 的大语言模型 (LLM)。其灵感来源于 Sebastian Raschka 的《Build a Large Language Model (From Scratch)》一书，旨在演示使用 PyTorch 构建和训练 LLM 的核心概念。

## ✨ 功能特性

- **模块化代码库**：核心组件（如注意力机制、Transformer 架构和数据加载器）被组织成独立的、易于理解的脚本。
- **多 GPU 训练**：支持使用 `torch.nn.DataParallel` 在多个 GPU 上进行训练，以加速训练过程。
- **日志记录**：集成了日志记录功能，用于监控训练进度和调试。
- **文本生成**：包含用于通过训练好的模型生成新文本的脚本，支持简单采样和温度缩放采样。
- **数据处理**：自定义的 PyTorch `Dataset` 和 `DataLoader`，用于高效处理文本数据。

## 📂 项目结构

项目的主要脚本位于 `scripts/` 目录下：

```
scripts/
├── GPTmodel.py              # 完整的 GPT 模型架构
├── train_model.py           # 模型训练的主脚本
├── generate_text.py         # 文本生成相关的工具函数
├── GPT_dataset_V1.py        # 自定义 PyTorch Dataset
├── utils.py                 # 共享的工具函数
├── attention*.py            # 各种注意力机制的实现
└── ...                      # 其他 Transformer 模型的核心构建模块
```

训练数据应放置在 `dataset/` 目录下。

## 🚀 快速开始

### 先决条件

- Python 3.8+
- PyTorch
- `requirements.txt` 中列出的所有包

### 安装

1.  **克隆仓库：**
    ```bash
    git clone https://github.com/your-username/LLM-From-Scratch.git
    cd LLM-From-Scratch
    ```

2.  **安装依赖：**
    建议在虚拟环境中使用。
    ```bash
    pip install -r requirements.txt
    ```

### 如何运行

#### 1. 准备数据集

在开始训练之前，请确保 `dataset/` 目录下包含您的训练数据文件（例如 `the-verdict.txt`）。如果目录为空，您可以从网络上下载一个示例文本文件。

#### 2. 训练模型

运行 `train_model.py` 脚本来开始训练：

```bash
python scripts/train_model.py
```

- **多 GPU 支持**：脚本会自动检测并使用所有可用的 GPU。
- **日志输出**：训练进度将通过日志打印到控制台。
- **训练结果**：训练和验证损失的变化曲线图将被保存为 `training_loss_plot.png`。

#### 3. 生成文本

模型训练完成后，`train_model.py` 脚本会自动调用文本生成功能来展示模型的效果。您也可以直接运行 `generate_text.py` 来查看其提供的工具函数。

## 🤝 贡献

我们非常欢迎各种形式的贡献！如果您想为这个项目做出贡献，请查阅我们的 [贡献指南](./CONTRIBUTING.md)。

## 📜 行为准则

为了确保一个健康和包容的社区环境，我们期望所有参与者都能遵守我们的 [行为准则](./CODE_OF_CONDUCT.md)。

## 🙏 致谢

本项目的灵感和部分代码结构来源于 Sebastian Raschka 的书籍 [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)。我们对此表示衷心的感谢。

## 📄 许可证

本项目采用 MIT 许可证。详情请见 [LICENSE](./LICENSE) 文件。