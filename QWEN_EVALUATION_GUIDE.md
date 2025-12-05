# 使用 Qwen 模型評估回答質量指南

## 📋 概述

本專案使用 **Qwen 模型**作為評估器（Judge），來比較微調後的 BERT 模型和原始模型的回答質量。這是一種 **LLM-as-a-Judge** 的評估方法。

## 🎯 核心原理

**EvalLLm 本質上就是通過 Prompt 讓 LLM 進行評價**

這是一種**元評估（Meta-Evaluation）**方法：
- 使用一個 LLM（Qwen）來評估另一個 LLM（BERT）的回答質量
- 通過精心設計的 Prompt 引導 Qwen 扮演"法官"角色
- Qwen 根據 Prompt 中的評估標準給出評分和判斷

## 🔍 工作原理

### 1. 評估流程（Prompt-based Evaluation）

```
問題 + 兩個模型的回答
    ↓
構建評估 Prompt（包含問題、標準答案、兩個模型的回答 + 評估指令）
    ↓
將 Prompt 輸入給 Qwen 模型
    ↓
Qwen 模型根據 Prompt 中的指令生成評估文本
    ↓
從 Qwen 的輸出中解析分數
    ↓
計算準確率和平均分數
```

### 2. 本質：Prompt Engineering

**關鍵點：**
- ✅ **不是程式化的評估**：不是用 if-else 或規則來判斷
- ✅ **是 Prompt 引導的 LLM 生成**：通過自然語言指令讓 Qwen 理解任務
- ✅ **依賴 LLM 的理解能力**：Qwen 需要理解問題、答案、評估標準
- ✅ **結果是文本生成**：Qwen 生成評估文本，然後我們解析分數

**類比：**
就像請一個人類專家來評估，只不過這個"專家"是 Qwen 模型，而我們通過 Prompt 來"告訴"它如何評估。

### 2. EvalLLm 類說明

**類定義：**
```python
class EvalLLm:
    def __init__(self,
            model_a_resp=None,      # 模型 A 的回答（原始模型）
            model_b_resp=None,      # 模型 B 的回答（微調模型）
            ground_truth=None,      # 標準答案
            prompt:str=None,        # 問題文本
            model_name:str="Qwen/Qwen2.5-3B-Instruct",  # 評估模型
            device:str=None):        # 設備
```

**核心方法：**

#### `evaluation(question: str)`
- **功能**：使用 Qwen 模型評估兩個回答的質量
- **輸入**：問題文本
- **輸出**：評估分數列表（如 [8, 9] 表示 Model A: 8分, Model B: 9分）

#### `judgement(scores: list, full_score: int=10)`
- **功能**：根據分數列表計算統計信息
- **輸入**：分數列表
- **輸出**：準確率和平均分數

## 📝 評估 Prompt 結構

### Prompt 的組成部分

一個完整的評估 Prompt 包含：

1. **角色設定**：`"Act as an impartial judge"` - 讓 Qwen 扮演公正的法官
2. **輸入信息**：問題、兩個模型的回答、標準答案（可選）
3. **評估指令**：告訴 Qwen 如何評估（正確性、相關性、完整性）
4. **輸出格式要求**：要求給出分數和判斷

### 有標準答案時（EVAL_PROMPT）

```
[Question]: {question}
[Reference Answer]: {gold_answer}        ← 提供標準答案作為參考
[Model A Answer]: {baseline_output}     ← 模型 A 的回答
[Model B Answer]: {finetuned_output}    ← 模型 B 的回答

Act as an impartial judge.              ← 角色設定
Evaluate both Model A and Model B's answers with respect to correctness, relevance, and completeness.
Give a score from 1-10 for each, and declare which is better.
```

### 無標準答案時

```
[Question]: {question}
[Model A Answer]: {baseline_output}
[Model B Answer]: {finetuned_output}

Act as an impartial judge.
Evaluate both Model A and Model B's answers with respect to correctness, relevance, and completeness.
Give a score from 1-10 for each, and declare which is better.
```

### Prompt 如何工作？

**步驟 1：構建 Prompt**
```python
eval_prompt = f"""
[Question]: 最大的行星是？
[Model A Answer]: 地球
[Model B Answer]: 木星

Act as an impartial judge.
Evaluate both Model A and Model B's answers...
"""
```

**步驟 2：輸入給 Qwen**
```python
messages = [
    {"role": "system", "content": "You are an impartial judge."},
    {"role": "user", "content": eval_prompt}
]
```

**步驟 3：Qwen 生成評估**
```
Qwen 的輸出可能是：
"Model A's answer (地球) is incorrect. Model B's answer (木星) is correct.
Model A: 2/10 (incorrect), Model B: 10/10 (correct and complete), Better: B"
```

**步驟 4：解析分數**
```python
# 從 Qwen 的文本輸出中提取分數
scores = [2, 10]  # Model A: 2分, Model B: 10分
```

## 🎯 評估標準

Qwen 模型會根據以下三個維度評估：

1. **正確性 (Correctness)**：答案是否正確
2. **相關性 (Relevance)**：答案是否與問題相關
3. **完整性 (Completeness)**：答案是否完整

每個維度評分 1-10 分，最終給出總體評分。

## 💻 使用範例

### 範例 1：基本使用

```python
from fine_tuning_llm_ipynb import EvalLLm

# 準備評估數據
question = "最大的行星是？"
model_a_answer = "地球"  # 原始模型回答
model_b_answer = "木星"  # 微調模型回答
ground_truth = "木星"    # 標準答案

# 創建評估器
eval_llm = EvalLLm(
    model_a_resp=model_a_answer,
    model_b_resp=model_b_answer,
    ground_truth=ground_truth,
    prompt=question
)

# 執行評估
scores = eval_llm.evaluation(question)
# 輸出: 模型回答（Qwen 的評估文本）
# 可能輸出: "Model A: 3/10, Model B: 10/10, Better: B"

# 計算統計信息
result = eval_llm.judgement(scores)
# 輸出: Accuracy: XX% | 平均信心分數: X.XX/10
```

### 範例 2：批量評估

```python
from fine_tuning_llm_ipynb import EvalLLm

# 測試問題列表
test_questions = [
    {
        "question": "最大的行星是？",
        "model_a": "地球",
        "model_b": "木星",
        "ground_truth": "木星"
    },
    {
        "question": "Python 是哪種語言？",
        "model_a": "編譯型",
        "model_b": "解釋型",
        "ground_truth": "解釋型"
    }
]

all_scores = []

for item in test_questions:
    eval_llm = EvalLLm(
        model_a_resp=item["model_a"],
        model_b_resp=item["model_b"],
        ground_truth=item["ground_truth"],
        prompt=item["question"]
    )
    
    scores = eval_llm.evaluation(item["question"])
    if scores:
        all_scores.extend(scores)

# 計算整體準確率
if all_scores:
    result = eval_llm.judgement(all_scores)
    print(f"整體評估結果: {result}")
```

### 範例 3：自訂評估模型

```python
# 使用其他 Qwen 模型
eval_llm = EvalLLm(
    model_a_resp="回答A",
    model_b_resp="回答B",
    ground_truth="標準答案",
    prompt="問題",
    model_name="Qwen/Qwen2.5-7B-Instruct",  # 使用更大的模型
    device="cuda"  # 指定設備
)
```

## 🔧 技術細節

### 1. 模型加載

```python
# 自動從 Hugging Face 下載 Qwen 模型
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
```

### 2. Prompt 構建與輸入

**關鍵：這不是程式化判斷，而是文本生成**

```python
# 步驟 1：構建 Prompt（自然語言文本）
eval_prompt = f"""
[Question]: {question}
[Model A Answer]: {model_a_answer}
[Model B Answer]: {model_b_answer}

Act as an impartial judge...
"""

# 步驟 2：包裝成對話格式
messages = [
    {"role": "system", "content": "You are an impartial judge."},
    {"role": "user", "content": eval_prompt}  # ← 這就是完整的評估指令
]

# 步驟 3：應用聊天模板（轉換成模型理解的格式）
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
# 輸出類似：
# "<|im_start|>system\nYou are an impartial judge...<|im_end|>\n<|im_start|>user\n[Question]: ...<|im_end|>\n<|im_start|>assistant\n"
```

### 3. 文本生成（LLM 生成評估）

**核心：Qwen 根據 Prompt 生成評估文本**

```python
# 將 Prompt 編碼成 token
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# Qwen 模型根據 Prompt 生成評估文本
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,  # 最大生成長度
    temperature=0.7,     # 溫度參數（控制隨機性）
    do_sample=True       # 啟用採樣
)

# 解碼生成的文本
response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
# 輸出可能是：
# "Model A's answer is incorrect. Model B's answer is correct. 
#  Model A: 2/10, Model B: 10/10, Better: B"
```

### 4. 分數解析（從文本中提取）

**注意：分數是從 Qwen 生成的文本中解析出來的，不是直接計算的**

```python
import re
# 嘗試從 Qwen 的文本輸出中提取分數
score_pattern = r'Model [AB]:\s*(\d+)/10'
found_scores = re.findall(score_pattern, response)

# 如果 Qwen 的輸出格式不標準，可能無法解析
if len(found_scores) >= 2:
    scores = [int(s) for s in found_scores[:2]]
else:
    # 需要手動查看 Qwen 的評估文本
    print("無法自動解析分數")
```

## 🎭 LLM-as-a-Judge 的本質

### 與傳統評估方法的對比

| 方法 | 評估方式 | 優點 | 缺點 |
|------|---------|------|------|
| **規則評估** | if-else、正則表達式 | 快速、可重現 | 無法處理複雜語義 |
| **程式化評估** | 計算相似度、BLEU等 | 客觀、可量化 | 無法理解語義 |
| **LLM-as-a-Judge** | Prompt + LLM 生成 | 理解語義、靈活 | 可能不一致、需要解析 |

### 為什麼使用 LLM 評估？

1. **語義理解**：LLM 能理解問題和答案的語義關係
2. **靈活性**：可以評估各種類型的問題，不需要預定義規則
3. **人類化評估**：更接近人類專家的評估方式
4. **可擴展性**：通過修改 Prompt 可以調整評估標準

### Prompt 的作用

**Prompt 是"指令"，告訴 LLM：**
- 你的角色是什麼（法官）
- 你要評估什麼（兩個回答）
- 如何評估（正確性、相關性、完整性）
- 輸出什麼格式（分數和判斷）

**類比：**
```
傳統方法：寫程式碼判斷 if answer == "木星": score = 10
LLM方法：寫 Prompt 告訴 Qwen "請評估這個答案是否正確，給出1-10分"
```

## ⚙️ 配置選項

### 可調整的參數

1. **評估模型**：
   - `Qwen/Qwen2.5-3B-Instruct`（預設，較小，速度快）
   - `Qwen/Qwen2.5-7B-Instruct`（更大，更準確）
   - `Qwen/Qwen2.5-14B-Instruct`（最大，最準確但需要更多資源）

2. **生成參數**：
   - `max_new_tokens`: 最大生成長度（預設 512）
   - `temperature`: 溫度參數（預設 0.7，越高越隨機）
   - `do_sample`: 是否啟用採樣（預設 True）

3. **評估標準**：
   - 可以修改 `EVAL_PROMPT` 來調整評估標準
   - 可以添加更多評估維度

## 📊 評估結果解讀

### 分數含義

- **10/10**：完美回答，完全正確、相關、完整
- **8-9/10**：優秀回答，基本正確但有輕微不足
- **6-7/10**：良好回答，部分正確但有一些問題
- **4-5/10**：一般回答，有明顯錯誤或不足
- **1-3/10**：差勁回答，嚴重錯誤或不相關

### 統計指標

- **準確率 (Accuracy)**：獲得滿分（10分）的比例
- **平均分數 (Average Score)**：所有評估的平均分數

## ⚠️ 注意事項

1. **首次使用**會下載 Qwen 模型（約 6-12 GB），需要較長時間
2. **記憶體需求**：Qwen2.5-3B 需要約 6-8GB RAM/VRAM
3. **評估時間**：每個問題的評估需要幾秒到幾十秒
4. **分數解析**：如果 Qwen 的回答格式不標準，可能無法自動解析分數
5. **網路連接**：需要網路連接以下載模型（首次使用）
6. **評估一致性**：LLM 評估可能每次略有不同（因為有隨機性）
7. **Prompt 依賴**：評估質量很大程度上依賴 Prompt 的設計

## 💡 理解 LLM-as-a-Judge 的本質

### 核心概念

**EvalLLm 本質上就是：**
1. **構建一個 Prompt**（包含問題、答案、評估指令）
2. **將 Prompt 輸入給 Qwen 模型**
3. **Qwen 根據 Prompt 生成評估文本**
4. **從生成的文本中解析分數**

**這不是：**
- ❌ 程式化的 if-else 判斷
- ❌ 計算相似度的數學方法
- ❌ 規則匹配的評估

**這是：**
- ✅ 通過自然語言 Prompt 引導 LLM
- ✅ 依賴 LLM 的語言理解和生成能力
- ✅ 類似於請人類專家評估，但專家是 LLM

### 實際流程示例

```python
# 1. 構建 Prompt（自然語言文本）
prompt = """
[Question]: 最大的行星是？
[Model A Answer]: 地球
[Model B Answer]: 木星

Act as an impartial judge. Evaluate both answers...
"""

# 2. 輸入給 Qwen（就像問人類專家一樣）
qwen_response = qwen_model.generate(prompt)
# Qwen 生成：
# "Model A's answer (地球) is incorrect. 
#  Model B's answer (木星) is correct.
#  Model A: 2/10, Model B: 10/10, Better: B"

# 3. 解析分數（從文本中提取）
scores = extract_scores(qwen_response)  # [2, 10]
```

### 為什麼這種方法有效？

1. **LLM 的語言理解能力**：Qwen 能理解問題和答案的語義
2. **上下文理解**：能考慮問題的完整上下文
3. **靈活判斷**：不需要預定義所有可能的評估規則
4. **接近人類評估**：評估方式更接近人類專家的判斷

### 局限性

1. **不一致性**：同樣的輸入可能產生略有不同的評估
2. **解析困難**：需要從自然語言文本中提取結構化信息
3. **成本**：需要運行一個大型 LLM（Qwen）
4. **依賴 Prompt**：評估質量很大程度上取決於 Prompt 設計

## 🔄 改進建議

### 1. 更精確的分數提取

可以改進正則表達式來更好地解析分數：

```python
# 更靈活的分數提取
score_patterns = [
    r'Model A[:\s]+(\d+)/10',
    r'Model B[:\s]+(\d+)/10',
    r'Score A[:\s]+(\d+)',
    r'Score B[:\s]+(\d+)'
]
```

### 2. 結構化輸出

可以要求 Qwen 使用 JSON 格式輸出，便於解析：

```python
eval_prompt = f"""
...（評估 prompt）...

Please respond in JSON format:
{{
    "model_a_score": X,
    "model_b_score": Y,
    "better": "A" or "B",
    "reason": "explanation"
}}
"""
```

### 3. 批量評估優化

可以將多個問題一起評估，提高效率：

```python
# 批量構建 prompts
batch_prompts = [build_eval_prompt(q) for q in questions]

# 批量生成
batch_outputs = model.generate(batch_prompts, ...)
```

## 📚 參考資料

- [Qwen 模型文檔](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [LLM-as-a-Judge 論文](https://arxiv.org/abs/2306.05685)
- [Transformers Chat Template](https://huggingface.co/docs/transformers/chat_templates)

## 🎯 實際應用場景

1. **模型對比**：比較微調前後的效果
2. **A/B 測試**：測試不同模型配置的效果
3. **質量監控**：監控模型在生產環境中的表現
4. **持續改進**：根據評估結果改進模型

---

**提示**：如果評估結果不理想，可以嘗試：
- 使用更大的 Qwen 模型
- 調整評估 prompt 的措辭
- 使用多個評估器並取平均分數

