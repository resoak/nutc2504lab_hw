import os
import pandas as pd
import requests
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

# === 基礎組件 ===
EMBED_URL = "https://ws-04.wade0426.me/embed"
COLLECTION_NAME = "day6_hw_final_run"
LOCAL_RERANKER_PATH = r"C:\Users\RS\Downloads\Qwen3-Reranker-0.6B"

class FastLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.client = OpenAI(api_key="No", base_url="https://ws-02.wade0426.me/v1")
    def load_model(self): return self.client
    def generate(self, prompt: str) -> str:
        res = self.client.chat.completions.create(model="google/gemma-3-27b-it", messages=[{"role": "user", "content": prompt}], temperature=0)
        return res.choices[0].message.content
    async def a_generate(self, prompt: str) -> str: return self.generate(prompt)
    def get_model_name(self): return "Gemma-3"

custom_llm = FastLLM()
q_client = QdrantClient(url="http://localhost:6333")
rerank_model = CrossEncoder(LOCAL_RERANKER_PATH, device="cuda", trust_remote_code=True)

def get_embeddings(texts: List[str]):
    return requests.post(EMBED_URL, json={"texts": texts, "normalize": True}).json()["embeddings"]

def advanced_search(query, bm25, all_chunks, top_k=5):
    q_vec = get_embeddings([query])[0]
    search_res = q_client.query_points(collection_name=COLLECTION_NAME, query=q_vec, limit=20).points
    bm25_scores = bm25.get_scores(query.split())
    top_bm25_idx = pd.Series(bm25_scores).nlargest(10).index
    candidates = list(set([h.payload["page_content"] for h in search_res] + [all_chunks[idx] for idx in top_bm25_idx]))
    pairs = [[query, cand] for cand in candidates]
    scores = rerank_model.predict(pairs, batch_size=1)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, s in ranked[:top_k]]

# 評估單一題目
def process_task(row, golden, bm25, all_chunks):
    qid, q_text = row['q_id'], row['questions']
    print(f"🚀 [Q{qid}] 啟動檢索生成...", flush=True)
    
    contexts = advanced_search(q_text, bm25, all_chunks)
    answer = custom_llm.generate(f"資訊：\n{''.join(contexts)}\n問題：{q_text}\n回答：")
    
    test_case = LLMTestCase(input=q_text, actual_output=answer, retrieval_context=contexts, expected_output=golden)
    
    # 關閉 async_mode=False 以便即時顯示 print
    metrics = [
        FaithfulnessMetric(model=custom_llm, async_mode=False),
        AnswerRelevancyMetric(model=custom_llm, async_mode=False),
        ContextualRecallMetric(model=custom_llm, async_mode=False),
        ContextualPrecisionMetric(model=custom_llm, async_mode=False),
        ContextualRelevancyMetric(model=custom_llm, async_mode=False)
    ]
    
    res = {"q_id": qid, "questions": q_text, "answer": answer}
    for m in metrics:
        m.measure(test_case)
        res[m.__class__.__name__] = m.score
    
    print(f"✅ [Q{qid}] 評估完畢 (Faithfulness: {res['FaithfulnessMetric']})", flush=True)
    return res

def main():
    # 準備資料
    with open("qa_data.txt", "r", encoding="utf-8") as f:
        all_chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(f.read())
    
    if q_client.collection_exists(COLLECTION_NAME): q_client.delete_collection(COLLECTION_NAME)
    q_client.create_collection(COLLECTION_NAME, vectors_config=models.VectorParams(size=len(get_embeddings(["t"])[0]), distance=models.Distance.COSINE))
    q_client.upsert(COLLECTION_NAME, points=[models.PointStruct(id=i, vector=v, payload={"page_content": t}) for i, (t, v) in enumerate(zip(all_chunks, get_embeddings(all_chunks)))])
    bm25 = BM25Okapi([doc.split() for doc in all_chunks])

    df_q = pd.read_csv("questions.csv")
    df_ans = pd.read_csv("questions_answer.csv")

    results = []
    # 使用 max_workers=2 兼顧速度與穩定性
    print(f"🔥 開始評估 30 題 (並行數: 2)...", flush=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for _, row in df_q.iterrows():
            golden = df_ans[df_ans['q_id'] == row['q_id']]['answer'].values[0]
            futures.append(executor.submit(process_task, row, golden, bm25, all_chunks))
        
        for f in futures:
            results.append(f.result())

    pd.DataFrame(results).sort_values("q_id").to_csv("day6_HW_questions.csv", index=False, encoding="utf-8-sig")
    print("\n🎉 恭喜！全部題目已處理完成。")

if __name__ == "__main__":
    main()