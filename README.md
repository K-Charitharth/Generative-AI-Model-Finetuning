# ⚡ Fast Fine-Tuning of Language Models using Unsloth for German-French Translation

![COMET Scores Comparison](/comet_comparision.png)

This project explores the efficient fine-tuning of a Large Language Model (LLM) using the [Unsloth](https://github.com/unslothai/unsloth) library. We focus on German-to-French translation using a benchmark dataset, with enhancements via data augmentation using a larger language model and evaluation through the COMET metric.

---

## 📌 Project Overview

-   **Goal:** Improve translation performance of a pre-trained LLM using efficient fine-tuning with Unsloth.
-   **Use Case:** German → French translation
-   **Highlights:**
    -   Fine-tuning with **QLoRA via Unsloth** for speed and memory efficiency.
    -   **Data augmentation** via synthetic translations (using Mistral 7B Instruct / Qwen2.5-Coder-32B-Instruct via Together.ai).
    -   Evaluation using the **COMET score** for robust translation quality assessment.
    -   **Google Colab-compatible**, enabling efficient training on a single GPU (T4 / A100).
    -   Interactive **Gradio UI** for real-time translation.

---

## 🧠 Key Technologies

| Component | Description |
| :---------------------- | :------------------------------------------------------------------------------------------------------ |
| **Unsloth** | Ultra-fast QLoRA fine-tuning framework for LLMs, offering significant speedups and memory savings. |
| **LLM (Base)** | Qwen2.5-1.5B-Instruct via HuggingFace Transformers, chosen for its balance of size and performance. |
| **LLM (Synthetic Data)** | Mistral 7B Instruct / Qwen2.5-Coder-32B-Instruct via Together.ai for high-quality synthetic data generation. |
| **Evaluation Metric** | [COMET](https://unbabel.github.io/COMET/) for advanced semantic-level translation quality assessment. |
| **Data Format** | Parallel sentence pairs (German-French) from OPUS-100 dataset. |
| **Platform** | Google Colab (T4 / A100 GPU) for accelerated training and development. |
| **User Interface** | [Gradio](https://www.gradio.app/) for an intuitive, interactive web application. |
| **Environment Management** | `python-dotenv` for secure handling of API keys. |

---

## 🚀 Getting Started

Follow these steps to set up and run the project.

### Prerequisites

Ensure you have Python 3.9+ installed.

### 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/german-french-translation.git](https://github.com/yourusername/german-french-translation.git)
    cd german-french-translation
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Generate `requirements.txt` using `pip freeze > requirements.txt` after installing all necessary libraries like `unsloth`, `datasets`, `unbabel-comet`, `gradio`, `trl`, `transformers`, `torch`, `requests`, `python-dotenv`, `matplotlib`).

### 🔑 API Key Setup

This project uses the **Together.ai API** for synthetic data generation.

1.  **Obtain an API Key:** Sign up on [Together.ai](https://www.together.ai/) and generate an API key.
2.  **Create a `.env` file:** In the root directory of the project, create a file named `.env` and add your API key:
    ```
    TOGETHER_API_KEY="your_together_ai_api_key_here"
    ```
    **Never commit your `.env` file to version control!** It's already added to `.gitignore` to prevent this.

### Project Structure
