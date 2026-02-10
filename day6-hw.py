import pandas as pd
import requests
import json
import time
import re
import os
from deepeval.metrics import (
    FaithfulnessMetric, 
    AnswerRelevancyMetric, 
    ContextualRecallMetric, 
    ContextualPrecisionMetric, 
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM

# --- 1. 配置與自定義模型 (強化 JSON 擷取穩定性) ---
API_URL = "https://ws-01.wade0426.me/v1/chat/completions"
MODEL_NAME = "allenai/olmOCR-2-7B-1025-FP8"

class CustomEvalModel(DeepEvalBaseLLM):
    def __init__(self, model_name):
        self.model_name = model_name
    def load_model(self):
        return self.model_name
    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are an evaluation judge. Respond ONLY with a JSON object. No prose."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0
        }
        try:
            response = requests.post(API_URL, json=payload, timeout=120)
            raw = response.json()['choices'][0]['message']['content']
            # 使用 Regex 強制擷取 JSON 區塊
            match = re.search(r'(\{.*\})', raw, re.DOTALL)
            return match.group(1) if match else raw
        except:
            return "{}"
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
    def get_model_name(self):
        return self.model_name

# --- 2. RAG 技術模組 (強化檢索召回率) ---

def llm_chat(prompt, system="你是一個專業的台水客服助手"):
    payload = {
        "model": MODEL_NAME, 
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}], 
        "temperature": 0.1
    }
    try:
        res = requests.post(API_URL, json=payload, timeout=60)
        return res.json()['choices'][0]['message']['content']
    except:
        return "無法生成答案"

def query_rewrite(q):
    """技術 1: Query Rewrite - 修正口語，產出核心關鍵字"""
    prompt = f"請將用戶問題改寫為 2-3 個適合搜尋的關鍵字（例如：簡訊帳單 申請 流程）：\n問題：{q}\n只需要回傳關鍵字，用空格隔開。"
    return llm_chat(prompt, "你負責優化搜尋關鍵字")

def hybrid_search(keywords, path='qa_data.txt'):
    """技術 2: Hybrid Search - 強化關鍵字匹配邏輯"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            corpus = f.readlines()
        
        word_list = keywords.split()
        results = []
        for line in corpus:
            line = line.strip()
            # 只要包含任何一個關鍵字就計分，分數越高排越前面
            score = sum(1 for word in word_list if word in line)
            if score > 0:
                results.append((score, line))
        
        # 按匹配程度排序
        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:3]] if results else ["目前資料庫無相關內容"]
    except:
        return ["讀取資料庫失敗"]

def rerank(q, ctx):
    """技術 3: Rerank - 從候選清單選出最精確的一段"""
    if len(ctx) <= 1 or ctx[0] == "目前資料庫無相關內容": return ctx
    prompt = f"問題：{q}\n搜尋結果：\n{ctx}\n請從中選出最能回答問題的『一段』原始文字回傳，不可修改內容。"
    return [llm_chat(prompt, "你是一個精準的重排篩選器")]

# --- 3. 執行邏輯 ---

def main():
    # 讀取檔案
    try:
        questions_df = pd.read_csv('questions.csv')
        truth_df = pd.read_csv('questions_answer.csv')
    except Exception as e:
        print(f"檔案讀取失敗: {e}")
        return

    # --- 第一階段：RAG 答案生成 ---
    print(">>> 階段一：生成 RAG 答案...")
    rag_temp_results = []

    for i, row in questions_df.iterrows():
        q_id, q_text = row['q_id'], row['questions']
        
        # RAG 鏈
        keywords = query_rewrite(q_text)
        contexts = hybrid_search(keywords)
        final_context = rerank(q_text, contexts)
        
        # 生成回答：加強約束避免 AI 亂編
        context_block = "\n".join(final_context)
        prompt = f"請嚴格根據參考資料回答。若資料沒寫，請說「抱歉，資料中無此資訊」。\n資料：{context_block}\n問題：{q_text}"
        answer = llm_chat(prompt)
        
        rag_temp_results.append({
            'q_id': q_id,
            'questions': q_text,
            'answer': answer,
            'contexts_json': json.dumps(final_context, ensure_ascii=False)
        })
        print(f"[{i+1}/{len(questions_df)}] 答案生成完成")

    # --- 第二階段：DeepEval 指標評測 ---
    print("\n>>> 階段二：DeepEval 指標評測中...")
    eval_model = CustomEvalModel(MODEL_NAME)
    
    metrics = {
        "F": FaithfulnessMetric(model=eval_model),
        "AR": AnswerRelevancyMetric(model=eval_model),
        "CR": ContextualRecallMetric(model=eval_model),
        "CP": ContextualPrecisionMetric(model=eval_model),
        "Crel": ContextualRelevancyMetric(model=eval_model)
    }

    final_output = []
    for row in rag_temp_results:
        q_id = row['q_id']
        ground_truth = truth_df[truth_df['q_id'] == q_id]['answer'].values[0]
        ctx_list = json.loads(row['contexts_json'])
        
        test_case = LLMTestCase(
            input=row['questions'],
            actual_output=row['answer'],
            retrieval_context=ctx_list,
            expected_output=ground_truth
        )
        
        scores = {}
        for k, m in metrics.items():
            try:
                m.measure(test_case)
                scores[k] = m.score
            except:
                scores[k] = 0
        
        final_output.append({
            'q_id': q_id,
            'questions': row['questions'],
            'answer': row['answer'],
            'Faithfulness（忠實度）': scores['F'],
            'Answer_Relevancy（答案相關性）': scores['AR'],
            'Contextual_Recall（上下文召回率）': scores['CR'],
            'Contextual_Precision（上下文精確度）': scores['CP'],
            'Contextual_Relevancy（上下文相關性）': scores['Crel']
        })
        print(f"Q_ID {q_id} 評測結束")

    # 儲存最終結果
    pd.DataFrame(final_output).to_csv('day6_HW_questions.csv', index=False, encoding='utf-8-sig')
    print("\n>>> 任務結束！請檢查 day6_HW_questions.csv")

if __name__ == "__main__":
    main()