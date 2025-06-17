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
    git clone https://github.com/K-Charitharth/Generative-AI-Model-Finetuning.git
    cd Generative-AI-Model-Finetuning
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

## 🔍 Insights

* **Unsloth's Efficiency:** Unsloth drastically reduced fine-tuning time while maintaining accuracy, making LLM adaptation feasible on consumer-grade GPUs.
* **Synthetic Data Impact:** Synthetic data from Mistral 7B / Qwen2.5-Coder-32B boosted generalization capabilities, especially useful in scenarios with limited real-world annotated data. It helps the model learn diverse linguistic patterns.
* **COMET vs. BLEU:** COMET proved effective for semantic-level evaluation, providing a more nuanced assessment of translation quality compared to traditional metrics like BLEU scores, which primarily focus on n-gram overlap.
* **Combined Data Advantage:** While Model B and D achieved the same score in this run, often combining diverse datasets (original + synthetic) can provide a more robust and generalized model. The slightly lower score for Model C might indicate that purely synthetic data, while beneficial, might lack the nuanced distribution of real data for this specific task compared to Model B.

---

## 🧠 Lessons Learned

The most interesting takeaway from this project was the use of **Unsloth**, which significantly reduced training time while allowing high-performance tuning of models as large as 7B parameters. Unlike traditional fine-tuning methods, Unsloth enables working on constrained hardware like Google Colab with remarkable speed, making LLM adaptation accessible to a broader range of developers. This opens doors for more rapid experimentation and iteration in specialized LLM applications.

---

## 📌 Future Work

* **Extend to multilingual translation:** Explore fine-tuning for more complex scenarios, e.g., DE-FR-EN triplets.
* **Optimize Synthetic Data Generation:** Experiment with different LLMs for synthetic data generation, diverse prompting strategies, and filtering techniques to improve synthetic data quality.
* **Explore LoRA Merging:** Investigate merging LoRA weights back into the base model for easier and more efficient model deployment in production environments.
* **Advanced Evaluation:** Incorporate human evaluation (if feasible) or more detailed error analysis beyond just COMET scores to identify specific areas of improvement.
* **Deployment Optimizations:** Explore quantization (beyond 4-bit) and distillation techniques for even smaller, faster deployment.

---

## ▶️ Demo Video

Watch a quick demonstration of the Gradio application in action:

[![Gradio App Demo](link_to_your_video_thumbnail.jpg)](/portfolio_interface.mp4)

---

## 🤝 Acknowledgements

* [Unsloth](https://github.com/unslothai/unsloth) for their incredible work on efficient fine-tuning.
* [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) for providing the foundation for LLM usage.
* [COMET Evaluation](https://unbabel.github.io/COMET/) for advanced translation metric.
* [Together.ai](https://www.together.ai/) for providing powerful LLMs for synthetic data generation.
* [OPUS Dataset](https://opus.nlpl.eu/) for the benchmark German-French parallel corpus.

---

## 📜 License

This project is open-source and distributed under the MIT License. See the `LICENSE` file for more details.
