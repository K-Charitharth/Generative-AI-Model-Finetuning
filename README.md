# ⚡ Fast Fine-Tuning of Language Models using Unsloth for German-French Translation

![COMET Scores Comparison](/comet_comparision.png)

This project demonstrates efficient fine-tuning of Large Language Models (LLMs) for German-to-French translation using the [Unsloth](https://github.com/unslothai/unsloth) library. The implementation features data augmentation through synthetic data generation and evaluates translation quality using the COMET metric. The solution achieves competitive translation quality with 4x faster training compared to traditional methods.

## 🧠 Key Features

- **Ultra-Fast Fine-tuning**: Uses QLoRA via Unsloth for 4x faster training
- **Data Augmentation**: Generates synthetic training data using Mistral 7B
- **Semantic Evaluation**: Utilizes COMET metric for translation quality assessment
- **Modular Architecture**: OOP-based design for maintainability and extensibility
- **Interactive Demo**: Gradio interface for real-time translations
- **Resource Efficiency**: Runs on a single GPU (T4/A100)

## 📊 Performance Comparison

| Model Version          | Training Data          | COMET Score | Training Time |
|------------------------|------------------------|-------------|---------------|
| Model A (Base)         | None                   | 0.65        | -             |
| Model B (Original)     | Original (800 samples) | 0.67        | 45 min        |
| Model C (Synthetic)    | Synthetic (1600 samples) | 0.65        | 60 min        |
| Model D (Combined)     | Combined (2400 samples) | 0.67        | 75 min        |

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- GPU with ≥16GB VRAM (NVIDIA T4/A100 recommended)
- Hugging Face account and API token
- Together API key (for synthetic data generation)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/german-french-translator.git
cd german-french-translator

# Install dependencies
pip install -r requirements.txt

# Install Unsloth (CUDA 12.1)
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
