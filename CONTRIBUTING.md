# 贡献指南

我们非常欢迎并感谢您对 LLM-From-Scratch 项目的兴趣和贡献！本指南将帮助您顺利地参与到项目中来。

## 如何贡献

我们欢迎各种形式的贡献，包括但不限于：
- 报告 Bug
- 提交功能建议
- 编写或改进文档
- 提交代码修复或新功能

## Bug 报告

如果您在项目中发现了 Bug，请通过 GitHub Issues 来报告。在提交时，请尽量提供以下信息：
- **清晰的标题**：简要描述问题。
- **复现步骤**：详细说明如何触发这个 Bug。
- **期望行为**：描述您认为应该发生什么。
- **实际行为**：描述实际发生了什么。
- **您的环境**：例如操作系统、Python 版本等。

## Pull Request (PR) 流程

1.  **Fork 仓库**：点击项目主页右上角的 "Fork" 按钮，将项目 Fork 到您自己的账户下。

2.  **克隆您的 Fork**：
    ```bash
    git clone https://github.com/YOUR_USERNAME/LLM-From-Scratch.git
    cd LLM-From-Scratch
    ```

3.  **创建新分支**：为您的修改创建一个新的分支。分支名称应能简要描述您的工作内容（例如 `fix-dataloader-bug` 或 `feature-add-attention-module`）。
    ```bash
    git checkout -b your-branch-name
    ```

4.  **进行修改**：在您的新分支上进行代码修改和开发。

5.  **提交您的修改**：
    ```bash
    git add .
    git commit -m "一个清晰、简洁的提交信息"
    ```

6.  **推送到您的 Fork**：
    ```bash
    git push origin your-branch-name
    ```

7.  **创建 Pull Request**：回到您在 GitHub 上的 Fork 仓库页面，点击 "New pull request" 按钮，创建一个 PR 到主项目的 `main` 分支。请在 PR 的描述中清晰地说明您的修改内容和目的。

## 代码风格

- **Python**: 请遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 编码规范。

感谢您的贡献！
