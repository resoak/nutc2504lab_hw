import os
import uuid
import pandas as pd
import requests
import re
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

# === 0. 配置與初始化 ===
API_KEY = "YOUR_API_KEY" # 請填入你的 API Key
EMBED_API_URL = "https://ws-04.wade0426.me/embed"
SUBMIT_URL = "https://hw-01.wade0426.me/submit_answer"

# 使用你指定的 ChatOpenAI 寫法
llm = ChatOpenAI(
    base_url="https://ws-05.huannago.com/v1",
    api_key=API_KEY, 
    model="google/gemma-3-27b-it",
    temperature=0 
)

CHUNK_SIZE = 300
CHUNK_OVERLAP = 120
client = QdrantClient(url="http://localhost:6333")

class CustomEmbeddings:
    def embed_documents(self, texts): return get_embeddings(texts)
    def embed_query(self, text): return get_embeddings([text])[0]

def get_embeddings(texts):
    if not texts: return []
    payload = {"texts": texts, "normalize": True, "batch_size": 32}
    try:
        response = requests.post(EMBED_API_URL, json=payload, timeout=60)
        return response.json()['embeddings']
    except: return []

def submit_and_get_score(q_id, answer):
    payload = {"q_id": q_id, "student_answer": answer}
    try:
        response = requests.post(SUBMIT_URL, json=payload, timeout=15)
        return response.json().get("score", 0)
    except: return 0

# === 1. 定義三種切分策略 ===

def get_all_chunks():
    data_files = [f"data_0{i}.txt" for i in range(1, 6)]
    full_content = ""
    for file_name in data_files:
        if os.path.exists(file_name):
            with open(file_name, "r", encoding="utf-8") as f:
                full_content += f.read() + "\n"
    
    print(f"📖 讀取資料完成，總字數: {len(full_content)}")
    
    # A. 固定大小
    fixed_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    fixed_chunks = fixed_splitter.split_text(full_content)
    
    # B. 滑動視窗
    sliding_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    sliding_chunks = sliding_splitter.split_text(full_content)
    
    # C. 語義切塊
    print("🧠 正在執行語義切塊 (這可能需要一點時間)...")
    sem_splitter = SemanticChunker(CustomEmbeddings(), breakpoint_threshold_type="percentile", breakpoint_threshold_amount=20)
    sem_docs = sem_splitter.create_documents([full_content])
    sem_chunks = [doc.page_content for doc in sem_docs]
    
    return {
        "固定大小": fixed_chunks,
        "滑動視窗": sliding_chunks,
        "語義切塊": sem_chunks
    }

# === 2. 執行 RAG 流程 ===

def run_experiment():
    methods_chunks = get_all_chunks()
    questions_df = pd.read_csv("questions.csv")
    q_texts = questions_df['questions'].tolist()
    q_ids = questions_df['q_id'].tolist()
    
    all_results = []

    for method_name, chunks in methods_chunks.items():
        print(f"\n🚀 開始評測方法: [{method_name}] (總塊數: {len(chunks)})")
        
        # 建立 Vector DB
        coll_name = f"coll_{uuid.uuid4().hex[:8]}"
        vectors = get_embeddings(chunks)
        client.recreate_collection(
            collection_name=coll_name,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
        )
        points = [PointStruct(id=i, vector=vectors[i], payload={"text": chunks[i], "idx": i}) for i in range(len(chunks))]
        client.upsert(collection_name=coll_name, points=points)

        method_scores = []
        for i, q_text in enumerate(q_texts):
            # 1. 檢索
            q_vec = get_embeddings([q_text])[0]
            hits = client.query_points(collection_name=coll_name, query=q_vec, limit=4).points
            hits.sort(key=lambda x: x.payload['idx']) # 按原文順序排序
            context = "\n".join([h.payload['text'] for h in hits])
            
            # 2. LLM 生成
            prompt = f"請根據以下資料回答問題。資料：\n{context}\n\n問題：{q_text}\n\n請給出簡潔精確的回答："
            response = llm.invoke(prompt)
            answer = response.content
            
            # 3. 提交並得分
            score = submit_and_get_score(q_ids[i], answer)
            method_scores.append(score)
            
            if (i+1) % 5 == 0:
                print(f"  - 已處理 {i+1}/{len(q_texts)} 題，當前進度平均分: {sum(method_scores)/len(method_scores):.4f}")
        
        avg_score = sum(method_scores) / len(method_scores)
        all_results.append({"method": method_name, "score": avg_score})
        print(f"✅ [{method_name}] 測試完成！平均分: {avg_score:.4f}")

    return pd.DataFrame(all_results)

if __name__ == "__main__":
    final_df = run_experiment()
    
    print("\n" + "="*50)
    print("📊 最終三種切法平均分統計表")
    print("="*50)
    print(final_df.sort_values(by="score", ascending=False).to_string(index=False))
    print("="*50)