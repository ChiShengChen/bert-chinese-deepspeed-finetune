#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
微調模型的推理腳本
用於載入訓練好的模型進行問答推理
"""

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import argparse
import os

def load_model(model_path, device=None):
    """
    載入微調後的模型和 tokenizer
    
    Args:
        model_path: 模型路徑
        device: 設備 (None 時自動檢測)
    
    Returns:
        model, tokenizer, device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"📦 正在載入模型從: {model_path}")
    print(f"🖥️  使用設備: {device}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路徑不存在: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        model.to(device)
        model.eval()
        print("✅ 模型載入成功！\n")
        return model, tokenizer, device
    except Exception as e:
        raise Exception(f"載入模型失敗: {e}")


def predict_mask(model, tokenizer, prompt, device, top_k=5):
    """
    使用 BERT 模型預測 [MASK] 位置的詞彙
    
    Args:
        model: 載入的模型
        tokenizer: tokenizer
        prompt: 輸入文本（會自動添加 [MASK]）
        device: 設備
        top_k: 返回前 k 個預測結果
    
    Returns:
        list: 預測的詞彙列表
    """
    # 確保 prompt 包含 [MASK]
    if tokenizer.mask_token not in prompt:
        prompt = prompt + tokenizer.mask_token
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    
    if len(mask_token_index) == 0:
        return ["⚠️ 未找到 [MASK] token"]
    
    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # 獲取 [MASK] 位置的 logits
    mask_token_logits = logits[0, mask_token_index[0], :]
    
    # 獲取 top_k 預測
    top_k_ids = torch.topk(mask_token_logits, top_k, dim=0).indices.tolist()
    predicted_tokens = [tokenizer.decode([idx]).strip() for idx in top_k_ids]
    
    return predicted_tokens


def qa_inference(model, tokenizer, question, options, device, top_k=3):
    """
    問答推理：根據問題和選項，預測最可能的答案
    
    Args:
        model: 載入的模型
        tokenizer: tokenizer
        question: 問題文本
        options: 選項字典，格式如 {"A": "選項A", "B": "選項B", ...}
        device: 設備
        top_k: 返回前 k 個最可能的答案
    
    Returns:
        list: 最可能的答案列表（按概率排序）
    """
    results = []
    
    for label, option_text in options.items():
        # 構建 prompt: 問題 + [MASK] + 選項
        prompt = f"{question} {tokenizer.mask_token} {option_text}"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        
        if len(mask_token_index) == 0:
            continue
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # 獲取 [MASK] 位置的 logits
        mask_logits = logits[0, mask_token_index[0], :]
        
        # 計算選項文本的 token 的平均分數
        option_tokens = tokenizer.encode(option_text, add_special_tokens=False)
        if len(option_tokens) > 0:
            # 取第一個 token 的分數作為代表
            score = mask_logits[option_tokens[0]].item()
            results.append((label, option_text, score))
    
    # 按分數排序
    results.sort(key=lambda x: x[2], reverse=True)
    
    return results[:top_k]


def interactive_mode(model, tokenizer, device):
    """
    互動模式：持續接收用戶輸入並進行推理
    """
    print("\n" + "="*60)
    print("🤖 BERT 微調模型推理模式")
    print("="*60)
    print("提示：")
    print("  - 輸入問題文本，會自動在末尾添加 [MASK]")
    print("  - 輸入 'quit' 或 'exit' 退出")
    print("  - 輸入 'qa' 進入問答模式")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("💬 請輸入問題: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 再見！")
                break
            
            if user_input.lower() == 'qa':
                # 問答模式
                question = input("❓ 問題: ").strip()
                print("📝 選項（格式：A:選項A B:選項B C:選項C D:選項D）")
                options_input = input("選項: ").strip()
                
                # 解析選項
                options = {}
                for opt in options_input.split():
                    if ':' in opt:
                        label, text = opt.split(':', 1)
                        options[label.strip()] = text.strip()
                
                if options:
                    results = qa_inference(model, tokenizer, question, options, device)
                    print("\n🎯 預測結果（按可能性排序）:")
                    for i, (label, text, score) in enumerate(results, 1):
                        print(f"  {i}. {label}: {text} (分數: {score:.4f})")
                print()
                continue
            
            # 標準 MASK 預測模式
            predictions = predict_mask(model, tokenizer, user_input, device, top_k=5)
            
            print("\n🔮 預測結果（Top 5）:")
            for i, pred in enumerate(predictions, 1):
                print(f"  {i}. {pred}")
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 再見！")
            break
        except Exception as e:
            print(f"❌ 錯誤: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="微調模型推理腳本")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./my_bert_finetuned_model_hf_format",
        help="模型路徑（預設: ./my_bert_finetuned_model_hf_format）"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="強制使用 CPU"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="單次推理模式：直接提供問題文本"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="返回前 k 個預測結果（預設: 5）"
    )
    
    args = parser.parse_args()
    
    # 設置設備
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 載入模型
    try:
        model, tokenizer, device = load_model(args.model_path, device)
    except Exception as e:
        print(f"❌ 無法載入模型: {e}")
        return
    
    # 單次推理模式
    if args.prompt:
        predictions = predict_mask(model, tokenizer, args.prompt, device, top_k=args.top_k)
        print(f"\n問題: {args.prompt}")
        print(f"\n預測結果（Top {args.top_k}）:")
        for i, pred in enumerate(predictions, 1):
            print(f"  {i}. {pred}")
    else:
        # 互動模式
        interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()

