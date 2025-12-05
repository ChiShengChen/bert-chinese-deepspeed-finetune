# BERT 中文模型微調專案

基於 DeepSpeed 框架的 BERT 中文模型微調專案，使用 TMMLU+ 多領域中文問答資料集進行訓練。

## 📋 專案簡介

本專案實現了使用 DeepSpeed 框架對 BERT 中文模型進行微調，支援在 40+ 個專業領域（醫學、法律、金融、物理等）的中文問答任務上進行訓練。專案從 Google Colab 遷移而來，已適配本地運行環境。

## ✨ 主要特性

- 🚀 **DeepSpeed 支援**：使用 DeepSpeed 框架進行高效訓練，支援 ZeRO 優化
- 🔄 **自動降級**：DeepSpeed 不可用時自動降級到標準 PyTorch 訓練
- 🎯 **多領域訓練**：涵蓋 40+ 個中文專業領域知識
- 💾 **模型自動保存**：訓練完成後自動保存模型到 `my_bert_finetuned_model_hf_format/`，可直接用於推理
- 📦 **檢查點管理**：支援訓練檢查點的保存和載入
- 📊 **視覺化**：自動生成訓練損失曲線圖
- 🔧 **設備自適應**：自動檢測並使用 GPU/CPU
- 📝 **完整評估**：包含模型評估和對比功能
- ⚠️ **重要說明**：BERT 是 Masked Language Model，不適合用於生成式聊天，適合填空和選擇題問答任務

## 🛠️ 環境要求

### 系統要求
- Python 3.8+
- CUDA 11.0+ (可選，用於 GPU 訓練)
- Linux / Windows / macOS

### 依賴庫

主要依賴：
- `torch` >= 1.9.0
- `transformers` >= 4.20.0
- `datasets` >= 2.0.0
- `deepspeed` >= 0.6.0 (可選)
- `matplotlib` >= 3.3.0
- `numpy` >= 1.20.0

## 📦 安裝步驟

### 1. 克隆或下載專案

```bash
cd /path/to/your/project
```

### 2. 建立虛擬環境（推薦）

```bash
conda create -n llm_finetune python=3.10
conda activate llm_finetune
```

或使用 venv：

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安裝依賴

```bash
# 安裝基礎依賴
pip install torch transformers datasets matplotlib numpy

# 安裝 DeepSpeed (可選，但推薦)
pip install deepspeed

# 如果需要 GPU 支援，根據 CUDA 版本安裝 PyTorch
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. 驗證安裝

```bash
python -c "import torch; import transformers; import deepspeed; print('✅ 所有依賴安裝成功')"
```

## 🚀 使用方法

### 基本運行

```bash
# 使用預設配置運行（自動檢測 GPU）
python fine_tuning_llm_ipynb.py

# 強制使用 CPU
python fine_tuning_llm_ipynb.py --cpu

# 指定檢查點保存路徑
python fine_tuning_llm_ipynb.py --save_dir ./my_checkpoints

# 從檢查點恢復訓練
python fine_tuning_llm_ipynb.py --load_dir ./checkpoints --ckpt_id step100
```

### 使用 DeepSpeed 啟動（推薦）

```bash
# 單 GPU
deepspeed fine_tuning_llm_ipynb.py

# 多 GPU
deepspeed --num_gpus=4 fine_tuning_llm_ipynb.py

# 使用配置檔案
deepspeed --deepspeed_config ds_config.json fine_tuning_llm_ipynb.py
```

### 使用微調後的模型進行推理

訓練完成後，可以使用 `inference.py` 腳本進行推理：

```bash
# 互動模式（推薦）
python inference.py

# 單次推理
python inference.py --prompt "今天天氣[MASK]"

# 指定模型路徑
python inference.py --model_path ./my_bert_finetuned_model_hf_format

# 強制使用 CPU
python inference.py --cpu

# 自訂返回結果數量
python inference.py --prompt "問題文本" --top_k 10
```

**互動模式功能：**
- 輸入問題文本，自動預測 [MASK] 位置的詞彙
- 輸入 `qa` 進入問答模式，可以比較多個選項
- 輸入 `quit` 或 `exit` 退出

## 🔍 inference.py 腳本詳細說明

### 腳本用途

`inference.py` 是一個獨立的推理腳本，用於載入訓練好的 BERT 模型並進行問答推理。它提供了兩種使用模式：**互動模式**和**單次推理模式**。

### 核心功能

#### 1. **模型載入** (`load_model`)

**功能：**
- 從指定路徑載入微調後的模型和 tokenizer
- 自動檢測並使用 GPU/CPU
- 將模型設置為評估模式（`model.eval()`）

**行為：**
```python
# 自動檢測設備
device = "cuda" if GPU可用 else "cpu"

# 載入模型
model = AutoModelForMaskedLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

#### 2. **填空預測** (`predict_mask`)

**功能：**
- 預測文本中 [MASK] 位置的詞彙
- 返回 top-k 個最可能的預測結果

**行為流程：**
```
輸入: "今天天氣很[MASK]"
  ↓
自動添加 [MASK]（如果沒有）
  ↓
Tokenize 輸入文本
  ↓
模型推理（獲取 logits）
  ↓
提取 [MASK] 位置的 logits
  ↓
選擇 top-k 個最高分數的 token
  ↓
輸出: ["好", "熱", "冷", "晴朗", "陰"]
```

**使用範例：**
```python
predictions = predict_mask(model, tokenizer, "人工智慧是[MASK]技術", device, top_k=5)
# 輸出: ["新興", "先進", "現代", "創新", "智能"]
```

#### 3. **問答推理** (`qa_inference`)

**功能：**
- 比較多個選項，找出最可能的答案
- 適用於選擇題問答場景

**行為流程：**
```
輸入: 
  問題: "最大的行星是？"
  選項: {"A": "地球", "B": "木星", "C": "火星"}
  ↓
對每個選項構建 prompt: "問題 [MASK] 選項"
  ↓
計算每個選項在 [MASK] 位置的得分
  ↓
按分數排序
  ↓
輸出: [("B", "木星", 8.5), ("A", "地球", 2.3), ("C", "火星", 1.1)]
```

**使用範例：**
```python
options = {"A": "地球", "B": "木星", "C": "火星"}
results = qa_inference(model, tokenizer, "最大的行星是？", options, device)
# 輸出: [("B", "木星", 8.5), ("A", "地球", 2.3), ...]
```

#### 4. **互動模式** (`interactive_mode`)

**功能：**
- 提供持續的互動式推理界面
- 支援兩種模式：填空預測和問答模式

**行為：**
```
啟動互動模式
  ↓
顯示使用提示
  ↓
等待用戶輸入
  ↓
判斷輸入類型：
  - "quit"/"exit" → 退出
  - "qa" → 進入問答模式
  - 其他 → 填空預測模式
  ↓
執行對應的推理
  ↓
顯示結果並繼續等待輸入
```

**互動模式範例：**
```
💬 請輸入問題: 今天天氣很[MASK]

🔮 預測結果（Top 5）:
  1. 好
  2. 熱
  3. 冷
  4. 晴朗
  5. 陰

💬 請輸入問題: qa
❓ 問題: 哪個是最大的行星？
📝 選項（格式：A:選項A B:選項B C:選項C D:選項D）
選項: A:地球 B:木星 C:火星 D:水星

🎯 預測結果（按可能性排序）:
  1. B: 木星 (分數: 8.5234)
  2. A: 地球 (分數: 2.3456)
  3. C: 火星 (分數: 1.1234)
```

### 使用模式

#### 模式 1：互動模式（預設）

```bash
python inference.py
```

**特點：**
- 持續運行，可以多次輸入問題
- 適合探索和測試模型
- 支援兩種推理方式（填空和問答）

#### 模式 2：單次推理模式

```bash
python inference.py --prompt "今天天氣[MASK]"
```

**特點：**
- 執行一次推理後退出
- 適合腳本自動化
- 可以配合其他工具使用

### 命令行參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--model_path` | 模型路徑 | `./my_bert_finetuned_model_hf_format` |
| `--cpu` | 強制使用 CPU | 自動檢測 |
| `--prompt` | 單次推理的問題文本 | `None`（互動模式） |
| `--top_k` | 返回前 k 個結果 | `5` |

### 程式碼結構

```
inference.py
├── load_model()          # 載入模型和 tokenizer
├── predict_mask()        # 填空預測功能
├── qa_inference()        # 問答推理功能
├── interactive_mode()    # 互動模式
└── main()                # 主函數（解析參數、啟動推理）
```

### 適用場景

**✅ 適合：**
- 測試微調後的模型效果
- 進行填空任務推理
- 選擇題問答系統
- 模型效果演示

**❌ 不適合：**
- 開放式對話（BERT 架構限制）
- 長文本生成
- 批量處理大量數據（建議使用程式碼 API）

### 與訓練腳本的關係

- **訓練腳本** (`fine_tuning_llm_ipynb.py`)：訓練模型並保存
- **推理腳本** (`inference.py`)：載入保存的模型進行推理
- 兩者分離，推理腳本可以獨立使用，無需重新訓練

## 📁 專案結構

```
LLM_example/
├── fine_tuning_llm_ipynb.py    # 主訓練腳本
├── inference.py                 # 模型推理腳本（用於載入模型進行問答）
├── checkpoints/                 # 訓練檢查點目錄（自動建立）
├── my_bert_finetuned_model_hf_format/  # 微調後的模型（訓練後生成）
├── test_qa_data.json            # 測試資料 JSON 檔案（自動生成）
├── validation_loss_curve.png   # 驗證損失曲線圖（自動生成）
└── README.md                    # 本檔案
```

## ⚙️ 配置說明

### DeepSpeed 配置

在程式碼中的 `config_params` 字典中可以調整訓練參數：

```python
config_params = {
    "train_batch_size": 32,              # 訓練批次大小
    "gradient_accumulation_steps": 1,     # 梯度累積步數
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-4,                   # 學習率
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
        "enabled": False                  # 混合精度訓練
    },
    "zero_optimization": {
        "stage": 0                        # ZeRO 優化階段 (0, 1, 2, 3)
    }
}
```

### 訓練參數

- `num_epochs`: 訓練輪數（預設 40）
- `save_interval`: 保存間隔步數（預設 20）
- `train_batch_size`: 批次大小（預設 32）
- `max_length`: 最大序列長度（預設 50）

### 資料集配置

程式碼支援從 TMMLU+ 資料集的多個領域載入資料，包括：
- 醫學、法律、金融、物理、化學等 40+ 個專業領域
- 自動按 70% / 25% / 5% 劃分訓練/驗證/測試集

## 📊 輸出檔案說明

### 1. 檢查點檔案 (`checkpoints/`)
訓練過程中保存的模型檢查點，可用於恢復訓練。

### 2. 微調模型 (`my_bert_finetuned_model_hf_format/`)
**訓練完成後自動保存的模型和 tokenizer，可直接用於推理。**

**模型保存功能：**
- ✅ 模型會自動保存到 `my_bert_finetuned_model_hf_format/` 目錄
- ✅ 包含完整的模型權重和 tokenizer 配置
- ✅ 使用 Hugging Face 格式，可直接用 `from_pretrained()` 載入
- ✅ 支援 DeepSpeed 和標準 PyTorch 兩種模式保存
- ✅ 程式碼中已包含載入和推理函數

**⚠️ 重要限制說明：**
- **BERT 是 Masked Language Model (MLM)**，不是生成式模型
- **不適合**用於開放式對話聊天（如 ChatGPT 那樣的連續對話）
- **適合**用於：
  - 填空任務：預測文本中 [MASK] 位置的詞彙
  - 選擇題問答：比較多個選項，找出最可能的答案
  - 文本理解和分類任務
- 如需真正的聊天功能，建議使用 **GPT 類生成式模型**（如 GPT-2、ChatGLM、Qwen 等）

### 3. 測試資料 (`test_qa_data.json`)
從測試集中提取的結構化問答資料。

### 4. 損失曲線 (`validation_loss_curve.png`)
訓練過程中的驗證損失視覺化圖表。

## 🔍 程式碼功能模組

### 1. 資料準備
- `generate_qa_benchmark()`: 生成問答基準測試資料
- `get_dataset()`: 載入和預處理多領域資料集

### 2. 模型訓練
- DeepSpeed 初始化（帶降級機制）
- 訓練循環（支援檢查點保存/載入）
- 驗證評估
- **自動保存微調後的模型**到 `my_bert_finetuned_model_hf_format/` 目錄

### 3. 模型保存與載入
- **模型保存**：訓練完成後自動保存模型和 tokenizer（第 403-442 行）
- **模型載入**：程式碼中包含載入函數（第 455-515 行）
- **推理功能**：提供 `chat_with_tuning_llm()` 函數進行推理
- **獨立推理腳本**：`inference.py` 提供更完整的推理功能

### 4. 模型推理
- `chat_with_tuning_llm()`: 使用微調模型進行推理（填空任務）
- `general_chat()`: 使用原始模型進行推理
- `inference.py`: 獨立的推理腳本，支援互動模式和問答模式

### 5. 模型評估
- `EvalLLm`: 使用 LLM 評估模型回答品質
- `exe_chat()`: 在測試集上執行完整評估流程

## 🐛 常見問題

### Q: DeepSpeed 初始化失敗怎麼辦？
A: 程式碼會自動降級到標準 PyTorch 訓練，無需擔心。如果想使用 DeepSpeed，請確保正確安裝：
```bash
pip install deepspeed
```

### Q: 記憶體不足怎麼辦？
A: 可以嘗試以下方法：
1. 減小 `train_batch_size`
2. 增加 `gradient_accumulation_steps`
3. 啟用 ZeRO Stage 2 或 3
4. 啟用 FP16 混合精度訓練

### Q: 如何調整訓練領域？
A: 修改 `get_dataset()` 函數中的 `task_list` 列表，新增或刪除需要的領域。

### Q: 模型保存失敗？
A: 檢查磁碟空間和寫入權限，確保有足夠的儲存空間。

### Q: CUDA out of memory 錯誤？
A: 
1. 減小批次大小
2. 使用梯度累積
3. 啟用 ZeRO 優化
4. 使用 CPU 訓練（新增 `--cpu` 參數）

## 📚 參考資料

- [DeepSpeed 官方文件](https://www.deepspeed.ai/)
- [Transformers 文件](https://huggingface.co/docs/transformers)
- [TMMLU+ 資料集](https://huggingface.co/datasets/ikala/tmmluplus)
- [BERT 中文模型](https://huggingface.co/bert-base-chinese)

## 📝 訓練流程說明

1. **資料載入**: 從 TMMLU+ 資料集載入多領域中文問答資料
2. **資料預處理**: Tokenization 和資料集劃分
3. **模型初始化**: 載入 BERT 中文預訓練模型
4. **DeepSpeed 初始化**: 配置訓練引擎（失敗則降級）
5. **訓練循環**: 
   - 前向傳播
   - 損失計算
   - 反向傳播
   - 參數更新
6. **驗證評估**: 每個 epoch 結束後在驗證集上評估
7. **模型保存**: **自動保存微調後的模型和 tokenizer 到 `my_bert_finetuned_model_hf_format/`**
8. **結果視覺化**: 生成損失曲線圖

## 💾 模型保存與使用

### 模型保存功能

**✅ 自動保存：**
- 訓練完成後，模型會自動保存到 `my_bert_finetuned_model_hf_format/` 目錄
- 保存格式為 Hugging Face Transformers 標準格式
- 包含完整的模型權重、配置檔案和 tokenizer

**✅ 保存內容：**
- `config.json`: 模型配置
- `pytorch_model.bin` 或 `model.safetensors`: 模型權重
- `tokenizer_config.json`: Tokenizer 配置
- `vocab.txt`: 詞彙表
- 其他必要的配置檔案

**✅ 載入方式：**
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

# 載入保存的模型
model_path = "./my_bert_finetuned_model_hf_format"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)
```

**✅ 程式碼中的載入和推理函數：**
- 程式碼中已包含載入和推理函數（第 455-515 行）
- 可以使用 `chat_with_tuning_llm()` 函數進行推理
- 建議使用 `inference.py` 腳本進行更完整的推理功能

### ⚠️ BERT 模型限制說明

**重要：BERT 不適合用於生成式聊天**

#### 為什麼 BERT Base Model 不適合 Chat？

**1. 架構設計差異：**

BERT 是 **雙向編碼器（Bidirectional Encoder）**：
- 使用 **Transformer Encoder** 架構
- 在訓練時可以同時看到整個序列的上下文（前後文）
- 設計目標是**理解**文本的語義，而非**生成**文本

生成式模型（如 GPT）是 **單向解碼器（Unidirectional Decoder）**：
- 使用 **Transformer Decoder** 架構
- 只能看到當前位置之前的上下文（因果遮罩）
- 設計目標是**自回歸生成**，逐個生成下一個 token

**2. 訓練目標不同：**

BERT 的訓練任務：
- **Masked Language Modeling (MLM)**：預測被遮罩的單個詞彙
- **Next Sentence Prediction (NSP)**：判斷兩個句子是否連續
- 目標是學習**雙向語義表示**

生成式模型的訓練任務：
- **Causal Language Modeling (CLM)**：根據前面的詞彙預測下一個詞彙
- 目標是學習**自回歸生成**能力

**3. 技術限制：**

BERT 的限制：
- ❌ **無法自回歸生成**：沒有解碼器的自注意力機制
- ❌ **無法處理序列生成**：只能預測單個 [MASK] 位置
- ❌ **沒有生成循環**：無法逐個生成 token 形成完整回答
- ❌ **雙向注意力不適合生成**：生成時不應該看到"未來"的信息

生成式模型的優勢：
- ✅ **自回歸生成**：可以逐個生成 token
- ✅ **序列生成能力**：可以生成任意長度的文本
- ✅ **因果遮罩**：確保生成時只使用已生成的內容

**4. 實際應用差異：**

BERT 的應用場景：
```
輸入: "今天天氣很[MASK]"
輸出: ["好", "熱", "冷", ...]  # 只能預測單個詞彙
```

生成式模型的應用場景：
```
輸入: "今天天氣很好，"
輸出: "今天天氣很好，適合出門散步。"  # 可以生成完整句子
```

**總結：**
- BERT 是**理解型模型**，專注於文本理解和語義表示
- GPT 類模型是**生成型模型**，專注於文本生成和對話
- 兩者的架構、訓練目標和應用場景完全不同
- 因此 BERT 不適合用於需要生成連續文本的聊天任務

**BERT 適合的任務：**
- ✅ 填空任務：`"今天天氣很[MASK]"` → 預測 "好"、"熱" 等
- ✅ 選擇題：比較多個選項，找出最可能的答案
- ✅ 文本分類：判斷文本類別
- ✅ 問答理解：理解問題和文本的語義關係

**不適合的任務：**
- ❌ 開放式對話：無法生成連續的對話文本
- ❌ 長文本生成：無法進行自回歸生成
- ❌ 創意寫作：無法進行自由創作
- ❌ 聊天機器人：無法像 ChatGPT 那樣進行多輪對話

**如需聊天功能，建議：**
- 使用 **GPT 類模型**（GPT-2、GPT-3、ChatGLM、Qwen 等）
- 使用 **Causal Language Model** 進行微調
- 本專案的 BERT 模型主要用於**問答理解**和**填空任務**

#### 技術對比表

| 特性 | BERT (Encoder) | GPT (Decoder) |
|------|----------------|----------------|
| **架構** | Transformer Encoder | Transformer Decoder |
| **注意力機制** | 雙向（Bidirectional） | 單向（Causal） |
| **訓練任務** | MLM + NSP | Causal LM |
| **生成能力** | ❌ 無法生成 | ✅ 可以生成 |
| **理解能力** | ✅ 優秀 | ✅ 良好 |
| **適合任務** | 分類、理解、填空 | 生成、對話、創作 |
| **聊天適用性** | ❌ 不適合 | ✅ 適合 |

#### 深入理解：為什麼 Encoder 架構不適合生成？

**核心問題：雙向注意力 vs 因果遮罩**

1. **BERT 的雙向注意力機制：**
   ```
   輸入序列: [CLS] 今天 天氣 很 [MASK] [SEP]
              ↑     ↑    ↑    ↑   ↑     ↑
   注意力:    所有位置都可以互相看到
   ```
   - 每個 token 可以看到整個序列（包括"未來"的 token）
   - 這在**理解任務**中很有用，因為可以同時考慮上下文
   - 但在**生成任務**中會造成問題：生成時不應該知道"未來"的內容

2. **GPT 的因果遮罩：**
   ```
   生成過程: "今天" → "天氣" → "很" → "好"
              ↑       ↑       ↑     ↑
   注意力:    只能看到已生成的內容（因果遮罩）
   ```
   - 每個 token 只能看到它**之前**的 token
   - 這確保生成過程是**自回歸**的：逐個生成，不依賴未來信息
   - 這是生成式模型的必要條件

3. **實際影響：**
   - BERT 無法實現因果遮罩，因為它的設計就是為了雙向理解
   - 即使強制使用 BERT 生成，也會因為看到"未來"信息而產生不一致的結果
   - 這就是為什麼需要專門的 Decoder 架構來進行文本生成

## 🎯 使用場景

- 中文問答系統開發（選擇題、填空題）
- 多領域知識理解任務
- 模型微調實驗
- 對比學習研究
- 中文 NLP 應用開發
- 文本填空和補全任務
- 問答系統的候選答案排序

## 💬 模型推理使用說明

### BERT 模型的適用場景

**✅ 適合：**
- 填空任務：`"今天天氣很[MASK]"` → 預測 "好"、"熱" 等
- 選擇題：比較多個選項，找出最可能的答案
- 文本理解：判斷文本的語義和意圖

**❌ 不適合：**
- 開放式對話：無法生成連續的對話文本
- 長文本生成：不是生成式模型
- 創意寫作：無法進行自由創作

### 推理範例

```python
# 使用 inference.py 進行推理

# 1. 填空任務
python inference.py --prompt "人工智慧是[MASK]技術"

# 2. 問答模式（互動模式中輸入 'qa'）
# 問題: 哪個是最大的行星？
# 選項: A:地球 B:木星 C:火星 D:水星
# → 模型會比較選項並給出最可能的答案
```

### 程式碼中使用

```python
from inference import load_model, predict_mask, qa_inference

# 載入模型
model, tokenizer, device = load_model("./my_bert_finetuned_model_hf_format")

# 填空預測
predictions = predict_mask(model, tokenizer, "今天天氣[MASK]", device, top_k=5)
print(predictions)  # ['好', '熱', '冷', '晴朗', '陰']

# 問答推理
options = {"A": "地球", "B": "木星", "C": "火星"}
results = qa_inference(model, tokenizer, "最大的行星是？", options, device)
print(results)  # [('B', '木星', 8.5), ('A', '地球', 2.3), ...]
```

## ⚠️ 注意事項

1. **首次運行**會下載預訓練模型和資料集，需要較長時間和網路連接
2. **訓練時間**取決於硬體配置，GPU 訓練會顯著加快速度
3. **儲存空間**：確保有足夠空間儲存模型和檢查點（約 1-2 GB）
4. **記憶體需求**：建議至少 8GB RAM，GPU 訓練需要 4GB+ 顯存
5. **BERT 模型限制**：
   - BERT 是 **Masked Language Model (MLM)**，不是生成式模型
   - **不適合**用於開放式對話聊天（如 ChatGPT）
   - **適合**用於：
     - 填空任務（預測 [MASK] 位置的詞彙）
     - 選擇題問答（比較選項的可能性）
     - 文本分類和理解任務
   - 如需真正的聊天功能，建議使用 **GPT** 類生成式模型

## 📄 授權許可

本專案基於原始 Colab notebook 修改，請參考原始專案的授權許可。

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📧 聯絡方式

如有問題或建議，請透過 Issue 回饋。

## 📖 完整使用範例

### 範例 1：完整訓練流程

```bash
# 1. 安裝依賴
pip install torch transformers datasets deepspeed matplotlib numpy

# 2. 開始訓練（使用 GPU）
python fine_tuning_llm_ipynb.py

# 訓練過程會顯示：
# - 資料載入進度
# - 每個 epoch 的訓練損失
# - 驗證損失
# - 模型自動保存到 my_bert_finetuned_model_hf_format/

# 3. 訓練完成後，使用模型進行推理
python inference.py
```

### 範例 2：使用 CPU 訓練

```bash
# 強制使用 CPU 訓練（適合沒有 GPU 的環境）
python fine_tuning_llm_ipynb.py --cpu

# 訓練時間會較長，但可以正常運行
```

### 範例 3：從檢查點恢復訓練

```bash
# 如果訓練中斷，可以從檢查點恢復
python fine_tuning_llm_ipynb.py \
    --load_dir ./checkpoints \
    --ckpt_id step100 \
    --save_dir ./checkpoints
```

### 範例 4：互動式推理

```bash
# 啟動互動模式
python inference.py

# 互動過程：
💬 請輸入問題: 人工智慧是[MASK]技術

🔮 預測結果（Top 5）:
  1. 新興
  2. 先進
  3. 現代
  4. 創新
  5. 智能

💬 請輸入問題: qa
❓ 問題: 哪個是最大的行星？
📝 選項（格式：A:選項A B:選項B C:選項C D:選項D）
選項: A:地球 B:木星 C:火星 D:水星

🎯 預測結果（按可能性排序）:
  1. B: 木星 (分數: 8.5234)
  2. A: 地球 (分數: 2.3456)
  3. C: 火星 (分數: 1.1234)

💬 請輸入問題: quit
👋 再見！
```

### 範例 5：單次推理（腳本模式）

```bash
# 直接提供問題，執行一次推理
python inference.py --prompt "今天天氣很[MASK]"

# 輸出：
# 問題: 今天天氣很[MASK]
# 
# 預測結果（Top 5）:
#   1. 好
#   2. 熱
#   3. 冷
#   4. 晴朗
#   5. 陰
```

### 範例 6：在 Python 程式碼中使用

```python
# 方法 1：使用 inference.py 的函數
from inference import load_model, predict_mask, qa_inference
import torch

# 載入模型
model, tokenizer, device = load_model("./my_bert_finetuned_model_hf_format")

# 填空預測
predictions = predict_mask(
    model, 
    tokenizer, 
    "人工智慧是[MASK]技術", 
    device, 
    top_k=5
)
print("預測結果:", predictions)
# 輸出: ['新興', '先進', '現代', '創新', '智能']

# 問答推理
options = {
    "A": "地球",
    "B": "木星", 
    "C": "火星",
    "D": "水星"
}
results = qa_inference(
    model,
    tokenizer,
    "最大的行星是？",
    options,
    device,
    top_k=3
)
print("問答結果:", results)
# 輸出: [('B', '木星', 8.5234), ('A', '地球', 2.3456), ...]
```

```python
# 方法 2：直接使用 Transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# 載入模型
model_path = "./my_bert_finetuned_model_hf_format"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 填空預測
prompt = "今天天氣很[MASK]"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    mask_token_logits = logits[0, mask_token_index[0], :]
    top_k_ids = torch.topk(mask_token_logits, 5, dim=0).indices.tolist()
    predictions = [tokenizer.decode([idx]).strip() for idx in top_k_ids]

print("預測結果:", predictions)
# 輸出: ['好', '熱', '冷', '晴朗', '陰']
```

### 範例 7：批量處理問題

```python
from inference import load_model, predict_mask

# 載入模型（只需載入一次）
model, tokenizer, device = load_model("./my_bert_finetuned_model_hf_format")

# 批量處理多個問題
questions = [
    "人工智慧是[MASK]技術",
    "深度學習是[MASK]的分支",
    "自然語言處理是[MASK]領域"
]

results = {}
for q in questions:
    predictions = predict_mask(model, tokenizer, q, device, top_k=3)
    results[q] = predictions[0]  # 取最可能的答案

print("批量處理結果:")
for question, answer in results.items():
    print(f"{question} → {answer}")
```

### 範例 8：使用 DeepSpeed 訓練

```bash
# 單 GPU 訓練
deepspeed fine_tuning_llm_ipynb.py

# 多 GPU 訓練（4 個 GPU）
deepspeed --num_gpus=4 fine_tuning_llm_ipynb.py

# 使用自訂配置檔案
deepspeed --deepspeed_config my_ds_config.json fine_tuning_llm_ipynb.py
```

### 範例 9：調整訓練參數

```python
# 在 fine_tuning_llm_ipynb.py 中修改 config_params
config_params = {
    "train_batch_size": 16,  # 減小批次大小（如果記憶體不足）
    "gradient_accumulation_steps": 2,  # 增加梯度累積
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 5e-5,  # 調整學習率
            "betas": [0.9, 0.999],
            "eps": 1e-9,
            "weight_decay": 3e-7
        }
    },
    "zero_optimization": {
        "stage": 2  # 使用 ZeRO Stage 2（節省記憶體）
    },
    "fp16": {
        "enabled": True  # 啟用混合精度訓練
    }
}

# 修改訓練輪數
num_epochs = 20  # 減少訓練輪數
```

### 範例 10：評估模型效果

```python
from inference import load_model, qa_inference

# 載入模型
model, tokenizer, device = load_model("./my_bert_finetuned_model_hf_format")

# 測試問題集
test_questions = [
    {
        "question": "最大的行星是？",
        "options": {"A": "地球", "B": "木星", "C": "火星"},
        "correct": "B"
    },
    {
        "question": "Python 是哪種語言？",
        "options": {"A": "編譯型", "B": "解釋型", "C": "機器語言"},
        "correct": "B"
    }
]

# 評估準確率
correct = 0
total = len(test_questions)

for item in test_questions:
    results = qa_inference(
        model, tokenizer, 
        item["question"], 
        item["options"], 
        device
    )
    predicted = results[0][0]  # 最可能的答案
    if predicted == item["correct"]:
        correct += 1
    print(f"問題: {item['question']}")
    print(f"預測: {predicted}, 正確: {item['correct']}")

accuracy = correct / total * 100
print(f"\n準確率: {accuracy:.2f}%")
```

### 範例 11：處理不同格式的輸入

```python
from inference import load_model, predict_mask

model, tokenizer, device = load_model("./my_bert_finetuned_model_hf_format")

# 情況 1：輸入已經包含 [MASK]
result1 = predict_mask(model, tokenizer, "今天[MASK]很好", device)
print("結果 1:", result1)

# 情況 2：輸入不包含 [MASK]（會自動添加）
result2 = predict_mask(model, tokenizer, "今天天氣很好", device)
print("結果 2:", result2)  # 會在末尾添加 [MASK]

# 情況 3：多個 [MASK]（只會預測第一個）
result3 = predict_mask(model, tokenizer, "[MASK]天氣很[MASK]", device)
print("結果 3:", result3)  # 只預測第一個 [MASK]
```

### 範例 12：保存推理結果

```python
from inference import load_model, predict_mask
import json

model, tokenizer, device = load_model("./my_bert_finetuned_model_hf_format")

# 準備問題列表
questions = [
    "人工智慧是[MASK]技術",
    "深度學習是[MASK]的分支",
    "機器學習是[MASK]的應用"
]

# 批量推理
results = []
for q in questions:
    predictions = predict_mask(model, tokenizer, q, device, top_k=3)
    results.append({
        "question": q,
        "predictions": predictions,
        "best_answer": predictions[0]
    })

# 保存結果
with open("inference_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("結果已保存到 inference_results.json")
```

---

**Happy Fine-tuning! 🚀**
