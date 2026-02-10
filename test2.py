import os
import pandas as pd
import requests
import torch
from typing import List
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from sentence_transformers import CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from deepeval.metrics import (
    FaithfulnessMetric, AnswerRelevancyMetric, 
    ContextualRecallMetric, ContextualPrecisionMetric, ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM

# === 1. 配置設定 ===
EMBED_URL = "https://ws-04.wade0426.me/embed"
COLLECTION_NAME = "day6_hw_final_run"
DEVICE = "cpu" 
LOCAL_RERANKER_PATH = os.path.expanduser("~/AI/Models/Qwen3-Reranker-0.6B")
TEMP_CSV = "rag_intermediate_results.csv"  
FINAL_CSV = "day6_HW_final_scores.csv"    

class FastLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.client = OpenAI(api_key="No", base_url="https://ws-05.huannago.com/v1")
    def load_model(self): return self.client
    def generate(self, prompt: str) -> str:
        res = self.client.chat.completions.create(
            model="google/gemma-3-27b-it", 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0
        )
        return res.choices[0].message.content
    async def a_generate(self, prompt: str) -> str: return self.generate(prompt)
    def get_model_name(self): return "Gemma-3"

custom_llm = FastLLM()
q_client = QdrantClient(url="http://localhost:6333")

# === 2. 載入模型 (含 Padding 修正) ===
print(f"🛠️  正在載入 Reranker (Device: {DEVICE})...")
try:
    rerank_model = CrossEncoder(LOCAL_RERANKER_PATH, device=DEVICE, trust_remote_code=True)
    if rerank_model.tokenizer.pad_token is None:
        rerank_model.tokenizer.pad_token = rerank_model.tokenizer.eos_token
        rerank_model.model.config.pad_token_id = rerank_model.tokenizer.eos_token_id
except Exception as e:
    print(f"⚠️  本地模型載入失敗: {e}")
    rerank_model = CrossEncoder("BAAI/bge-reranker-v2-m3", device=DEVICE)

# === 3. 檢索與動態維度處理 ===
def get_embeddings(texts: List[str]):
    res = requests.post(EMBED_URL, json={"texts": texts, "normalize": True})
    return res.json()["embeddings"]

def advanced_search(query, bm25, all_chunks, top_k=5):
    q_vec = get_embeddings([query])[0]
    search_res = q_client.query_points(collection_name=COLLECTION_NAME, query=q_vec, limit=20).points
    bm25_scores = bm25.get_scores(query.split())
    top_bm25_idx = pd.Series(bm25_scores).nlargest(10).index
    candidates = list(set([h.payload["page_content"] for h in search_res] + [all_chunks[idx] for idx in top_bm25_idx]))
    pairs = [[query, cand] for cand in candidates]
    # batch_size=1 確保 Qwen3 穩定
    scores = rerank_model.predict(pairs, batch_size=1, show_progress_bar=False)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, s in ranked[:top_k]]

# === 4. 主程式 ===
def main():
    # --- A. 準備資料與動態索引建立 ---
    print("📖 讀取文本與建立索引中...")
    with open("qa_data.txt", "r", encoding="utf-8") as f:
        all_chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(f.read())
    
    # 【動態維度偵測】
    print("🔍 偵測 Embedding 維度...")
    sample_embeddings = get_embeddings(["維度偵測測試內容"])
    vector_dim = len(sample_embeddings[0])
    print(f"✅ 偵測到向量維度為: {vector_dim}")

    if q_client.collection_exists(COLLECTION_NAME): 
        q_client.delete_collection(COLLECTION_NAME)
    
    q_client.create_collection(
        COLLECTION_NAME, 
        vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE)
    )

    # 批次寫入資料
    print("⚙️  正在將資料寫入向量庫...")
    chunk_embeddings = get_embeddings(all_chunks)
    q_client.upsert(
        COLLECTION_NAME, 
        points=[
            models.PointStruct(id=i, vector=v, payload={"page_content": t}) 
            for i, (t, v) in enumerate(zip(all_chunks, chunk_embeddings))
        ]
    )
    bm25 = BM25Okapi([doc.split() for doc in all_chunks])

    df_q = pd.read_csv("questions.csv")
    df_ans = pd.read_csv("questions_answer.csv")
    
    # --- B. 第一階段：生成 RAG 結果並存入 CSV ---
    print(f"\n🔥 階段 1：開始檢索與生成回答 (總計 {len(df_q)} 題)...")
    
    def run_rag(row):
        qid, q_text = row['q_id'], row['questions']
        contexts = advanced_search(q_text, bm25, all_chunks)
        answer = custom_llm.generate(f"資訊：\n{''.join(contexts)}\n問題：{q_text}\n回答：")
        golden = df_ans[df_ans['q_id'] == qid]['answer'].values[0]
        return {
            "q_id": qid, "input": q_text, "actual_output": answer, 
            "expected_output": golden, "retrieval_context": "|".join(contexts)
        }

    with ThreadPoolExecutor(max_workers=2) as executor:
        rag_results = list(executor.map(run_rag, [row for _, row in df_q.iterrows()]))

    pd.DataFrame(rag_results).to_csv(TEMP_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ 階段 1 完成，中間結果已存至 {TEMP_CSV}")

    # --- C. 第二階段：讀取 CSV 進行評估 ---
    print(f"\n🧐 階段 2：開始 DeepEval 品質評估指標分析...")
    df_eval = pd.read_csv(TEMP_CSV)
    
    metrics = [
        FaithfulnessMetric(model=custom_llm, async_mode=False),
        AnswerRelevancyMetric(model=custom_llm, async_mode=False),
        ContextualRecallMetric(model=custom_llm, async_mode=False),
        ContextualPrecisionMetric(model=custom_llm, async_mode=False),
        ContextualRelevancyMetric(model=custom_llm, async_mode=False)
    ]

    final_scores = []
    for _, row in df_eval.iterrows():
        print(f"正在評估 Q{row['q_id']}...")
        contexts = row['retrieval_context'].split("|")
        
        test_case = LLMTestCase(
            input=row['input'],
            actual_output=row['actual_output'],
            expected_output=row['expected_output'],
            retrieval_context=contexts
        )
        
        res_dict = row.to_dict()
        for m in metrics:
            m.measure(test_case)
            res_dict[m.__class__.__name__] = m.score
        
        final_scores.append(res_dict)

    # --- D. 儲存結果 ---
    pd.DataFrame(final_scores).sort_values("q_id").to_csv(FINAL_CSV, index=False, encoding="utf-8-sig")
    print(f"\n🎉 任務圓滿完成！最終評分報表：{FINAL_CSV}")

if __name__ == "__main__":
    main()