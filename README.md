# BERT Chinese Model Fine-tuning Project

**English** | [繁體中文](README_zh_TW.md)

---

A BERT Chinese model fine-tuning project based on DeepSpeed framework, trained on TMMLU+ multi-domain Chinese Q&A dataset.

## 📋 Project Overview

This project implements fine-tuning of BERT Chinese model using DeepSpeed framework, supporting training on 40+ professional domains (medicine, law, finance, physics, etc.) for Chinese Q&A tasks. The project was migrated from Google Colab and adapted for local execution.

## ✨ Key Features

- 🚀 **DeepSpeed Support**: Efficient training with DeepSpeed framework, supporting ZeRO optimization
- 🔄 **Auto Fallback**: Automatically falls back to standard PyTorch training when DeepSpeed is unavailable
- 🎯 **Multi-domain Training**: Covers 40+ Chinese professional domain knowledge
- 💾 **Auto Model Saving**: Automatically saves model to `my_bert_finetuned_model_hf_format/` after training, ready for inference
- 📦 **Checkpoint Management**: Supports saving and loading training checkpoints
- 📊 **Visualization**: Automatically generates training loss curve
- 🔧 **Device Adaptive**: Automatically detects and uses GPU/CPU
- 📝 **Complete Evaluation**: Includes model evaluation and comparison functions
- ⚠️ **Important Note**: BERT is a Masked Language Model, not suitable for generative chat, but suitable for fill-in-the-blank and multiple-choice Q&A tasks

## 🛠️ Requirements

### System Requirements
- Python 3.8+
- CUDA 11.0+ (optional, for GPU training)
- Linux / Windows / macOS

### Dependencies

Main dependencies:
- `torch` >= 1.9.0
- `transformers` >= 4.20.0
- `datasets` >= 2.0.0
- `deepspeed` >= 0.6.0 (optional)
- `matplotlib` >= 3.3.0
- `numpy` >= 1.20.0

## 📦 Installation

### 1. Clone or Download Project

```bash
cd /path/to/your/project
```

### 2. Create Virtual Environment (Recommended)

```bash
conda create -n llm_finetune python=3.10
conda activate llm_finetune
```

Or use venv:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
# Install basic dependencies
pip install torch transformers datasets matplotlib numpy

# Install DeepSpeed (optional but recommended)
pip install deepspeed

# For GPU support, install PyTorch according to CUDA version
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Verify Installation

```bash
python -c "import torch; import transformers; import deepspeed; print('✅ All dependencies installed successfully')"
```

## 🚀 Usage

### Basic Run

```bash
# Run with default configuration (auto-detect GPU)
python fine_tuning_llm_ipynb.py

# Force CPU usage
python fine_tuning_llm_ipynb.py --cpu

# Specify checkpoint save path
python fine_tuning_llm_ipynb.py --save_dir ./my_checkpoints

# Resume training from checkpoint
python fine_tuning_llm_ipynb.py --load_dir ./checkpoints --ckpt_id step100
```

### Using DeepSpeed (Recommended)

```bash
# Single GPU
deepspeed fine_tuning_llm_ipynb.py

# Multiple GPUs
deepspeed --num_gpus=4 fine_tuning_llm_ipynb.py

# With config file
deepspeed --deepspeed_config ds_config.json fine_tuning_llm_ipynb.py
```

### Using Fine-tuned Model for Inference

After training, use `inference.py` script for inference:

```bash
# Interactive mode (recommended)
python inference.py

# Single inference
python inference.py --prompt "今天天氣[MASK]"

# Specify model path
python inference.py --model_path ./my_bert_finetuned_model_hf_format

# Force CPU
python inference.py --cpu

# Custom top-k results
python inference.py --prompt "question text" --top_k 10
```

**Interactive Mode Features:**
- Input question text, automatically predicts [MASK] position vocabulary
- Input `qa` to enter Q&A mode, can compare multiple options
- Input `quit` or `exit` to exit

## 📁 Project Structure

```
LLM_example/
├── fine_tuning_llm_ipynb.py    # Main training script
├── inference.py                 # Model inference script
├── checkpoints/                 # Training checkpoint directory (auto-created)
├── my_bert_finetuned_model_hf_format/  # Fine-tuned model (generated after training)
├── test_qa_data.json            # Test data JSON file (auto-generated)
├── validation_loss_curve.png   # Validation loss curve (auto-generated)
└── README.md                    # This file
```

## ⚙️ Configuration

### DeepSpeed Configuration

Adjust training parameters in the `config_params` dictionary in code:

```python
config_params = {
    "train_batch_size": 32,              # Training batch size
    "gradient_accumulation_steps": 1,     # Gradient accumulation steps
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-4,                   # Learning rate
            "betas": [0.9, 0.999],
            "eps": 1e-9,
            "weight_decay": 3e-7
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-5,
            "warmup_num_steps": 100
        }
    },
    "fp16": {
        "enabled": False                  # Mixed precision training
    },
    "zero_optimization": {
        "stage": 0                        # ZeRO optimization stage (0, 1, 2, 3)
    }
}
```

### Training Parameters

- `num_epochs`: Number of training epochs (default: 40)
- `save_interval`: Save interval steps (default: 20)
- `train_batch_size`: Batch size (default: 32)
- `max_length`: Maximum sequence length (default: 50)

### Dataset Configuration

Code supports loading data from multiple domains of TMMLU+ dataset, including:
- Medicine, law, finance, physics, chemistry, and 40+ professional domains
- Automatically splits into 70% / 25% / 5% for train/validation/test sets

## 📊 Output Files

### 1. Checkpoint Files (`checkpoints/`)
Model checkpoints saved during training, can be used to resume training.

### 2. Fine-tuned Model (`my_bert_finetuned_model_hf_format/`)
**Automatically saved model and tokenizer after training, ready for inference.**

**Model Saving Features:**
- ✅ Model automatically saved to `my_bert_finetuned_model_hf_format/` directory
- ✅ Includes complete model weights and tokenizer configuration
- ✅ Uses Hugging Face format, can be loaded with `from_pretrained()`
- ✅ Supports both DeepSpeed and standard PyTorch modes
- ✅ Code includes loading and inference functions

**⚠️ Important Limitations:**
- **BERT is a Masked Language Model (MLM)**, not a generative model
- **Not suitable** for open-ended conversational chat (like ChatGPT)
- **Suitable** for:
  - Fill-in-the-blank tasks: Predict vocabulary at [MASK] position
  - Multiple-choice Q&A: Compare options to find most likely answer
  - Text understanding and classification tasks
- For true chat functionality, use **GPT-style generative models** (GPT-2, ChatGLM, Qwen, etc.)

### 3. Test Data (`test_qa_data.json`)
Structured Q&A data extracted from test set.

### 4. Loss Curve (`validation_loss_curve.png`)
Visualization chart of validation loss during training.

## 💾 Model Saving and Usage

### Model Saving Features

**✅ Auto Save:**
- After training, model is automatically saved to `my_bert_finetuned_model_hf_format/` directory
- Saved in Hugging Face Transformers standard format
- Includes complete model weights, configuration files, and tokenizer

**✅ Saved Contents:**
- `config.json`: Model configuration
- `pytorch_model.bin` or `model.safetensors`: Model weights
- `tokenizer_config.json`: Tokenizer configuration
- `vocab.txt`: Vocabulary
- Other necessary configuration files

**✅ Loading Method:**
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load saved model
model_path = "./my_bert_finetuned_model_hf_format"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)
```

### ⚠️ BERT Model Limitations

**Important: BERT is not suitable for generative chat**

#### Why BERT Base Model is Not Suitable for Chat?

**1. Architecture Differences:**

BERT is a **Bidirectional Encoder**:
- Uses **Transformer Encoder** architecture
- Can see entire sequence context (both forward and backward) during training
- Designed for **understanding** text semantics, not **generating** text

Generative models (like GPT) are **Unidirectional Decoders**:
- Use **Transformer Decoder** architecture
- Can only see context before current position (causal masking)
- Designed for **autoregressive generation**, generating next token one by one

**2. Different Training Objectives:**

BERT's training tasks:
- **Masked Language Modeling (MLM)**: Predict single masked vocabulary
- **Next Sentence Prediction (NSP)**: Judge if two sentences are consecutive
- Goal: Learn **bidirectional semantic representations**

Generative model's training tasks:
- **Causal Language Modeling (CLM)**: Predict next vocabulary based on previous ones
- Goal: Learn **autoregressive generation** capability

**3. Technical Limitations:**

BERT's limitations:
- ❌ **Cannot autoregressively generate**: No decoder self-attention mechanism
- ❌ **Cannot handle sequence generation**: Can only predict single [MASK] position
- ❌ **No generation loop**: Cannot generate tokens one by one to form complete answer
- ❌ **Bidirectional attention unsuitable for generation**: Should not see "future" information during generation

Generative model's advantages:
- ✅ **Autoregressive generation**: Can generate tokens one by one
- ✅ **Sequence generation capability**: Can generate text of arbitrary length
- ✅ **Causal masking**: Ensures only using already generated content during generation

**4. Practical Application Differences:**

BERT's application scenarios:
```
Input: "今天天氣很[MASK]"
Output: ["好", "熱", "冷", ...]  # Can only predict single vocabulary
```

Generative model's application scenarios:
```
Input: "今天天氣很好，"
Output: "今天天氣很好，適合出門散步。"  # Can generate complete sentence
```

**Summary:**
- BERT is a **understanding model**, focused on text understanding and semantic representation
- GPT-style models are **generative models**, focused on text generation and conversation
- Their architectures, training objectives, and application scenarios are completely different
- Therefore BERT is not suitable for chat tasks requiring continuous text generation

**BERT Suitable Tasks:**
- ✅ Fill-in-the-blank: `"今天天氣很[MASK]"` → Predict "好", "熱", etc.
- ✅ Multiple-choice: Compare multiple options to find most likely answer
- ✅ Text classification: Judge text category
- ✅ Q&A understanding: Understand semantic relationship between question and text

**Unsuitable Tasks:**
- ❌ Open-ended dialogue: Cannot generate continuous dialogue text
- ❌ Long text generation: Cannot autoregressively generate
- ❌ Creative writing: Cannot freely create
- ❌ Chatbot: Cannot have multi-turn conversations like ChatGPT

**For Chat Functionality, Recommended:**
- Use **GPT-style models** (GPT-2, GPT-3, ChatGLM, Qwen, etc.)
- Use **Causal Language Model** for fine-tuning
- This project's BERT model is mainly for **Q&A understanding** and **fill-in-the-blank tasks**

#### Technical Comparison Table

| Feature | BERT (Encoder) | GPT (Decoder) |
|---------|----------------|----------------|
| **Architecture** | Transformer Encoder | Transformer Decoder |
| **Attention Mechanism** | Bidirectional | Causal (Unidirectional) |
| **Training Task** | MLM + NSP | Causal LM |
| **Generation Capability** | ❌ Cannot generate | ✅ Can generate |
| **Understanding Capability** | ✅ Excellent | ✅ Good |
| **Suitable Tasks** | Classification, Understanding, Fill-in-blank | Generation, Dialogue, Creation |
| **Chat Suitability** | ❌ Not suitable | ✅ Suitable |

## 🐛 FAQ

### Q: What if DeepSpeed initialization fails?
A: Code will automatically fall back to standard PyTorch training, no worries. To use DeepSpeed, ensure proper installation:
```bash
pip install deepspeed
```

### Q: What if running out of memory?
A: Try the following methods:
1. Reduce `train_batch_size`
2. Increase `gradient_accumulation_steps`
3. Enable ZeRO Stage 2 or 3
4. Enable FP16 mixed precision training

### Q: How to adjust training domains?
A: Modify the `task_list` in `get_dataset()` function, add or remove needed domains.

### Q: Model save failed?
A: Check disk space and write permissions, ensure sufficient storage space.

### Q: CUDA out of memory error?
A: 
1. Reduce batch size
2. Use gradient accumulation
3. Enable ZeRO optimization
4. Use CPU training (add `--cpu` parameter)

## 📚 References

- [DeepSpeed Official Documentation](https://www.deepspeed.ai/)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [TMMLU+ Dataset](https://huggingface.co/datasets/ikala/tmmluplus)
- [BERT Chinese Model](https://huggingface.co/bert-base-chinese)

## ⚠️ Notes

1. **First run** will download pretrained models and datasets, requires longer time and network connection
2. **Training time** depends on hardware configuration, GPU training significantly speeds up
3. **Storage space**: Ensure sufficient space for models and checkpoints (approximately 1-2 GB)
4. **Memory requirements**: Recommend at least 8GB RAM, GPU training needs 4GB+ VRAM
5. **BERT Model Limitations**:
   - BERT is a **Masked Language Model (MLM)**, not a generative model
   - **Not suitable** for open-ended conversational chat (like ChatGPT)
   - **Suitable** for:
     - Fill-in-the-blank tasks (predict vocabulary at [MASK] position)
     - Multiple-choice Q&A (compare option probabilities)
     - Text classification and understanding tasks
   - For true chat functionality, use **GPT-style generative models**

## 📄 License

This project is modified from original Colab notebook, please refer to original project's license.

## 🤝 Contributing

Welcome to submit Issues and Pull Requests!

## 📧 Contact

For questions or suggestions, please provide feedback through Issues.

---

**Happy Fine-tuning! 🚀**
