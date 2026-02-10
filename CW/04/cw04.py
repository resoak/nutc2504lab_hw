import os
import glob
import pandas as pd
import uuid
import requests
import torch
import torch.nn.functional as F
from typing import List

# LangChain / OpenAI
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Qdrant & Transformers
from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer, AutoModelForCausalLM

# === 1. 配置與初始化 ===
VLM_BASE_URL = "https://ws-02.wade0426.me/v1"
VLM_MODEL = "google/gemma-3-27b-it"
EMBED_URL = "https://ws-04.wade0426.me/embed"
COLLECTION_NAME = "gemma_hybrid_qwen3_rerank"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

llm = ChatOpenAI(
    base_url=VLM_BASE_URL,
    api_key="YOUR_API_KEY", # ⚠️ 請填入您的 API Key
    model=VLM_MODEL,
    temperature=0
)

# === 2. 載入本地 Qwen3 Reranker (CausalLM 版) ===
model_path = os.path.expanduser("C:\\Users\\RS\Downloads\\Qwen3-Reranker-0.6B")
print(f"📦 正在載入 CausalLM Reranker: {model_path}")

reranker_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
reranker_model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map=DEVICE
).eval()

token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")
token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")

client = QdrantClient(url="http://localhost:6333")

# === 3. 工具函數 ===

def get_embeddings(texts: List[str]) -> List[List[float]]:
    payload = {"texts": texts, "normalize": True, "task_description": "檢索技術與生活文件"}
    try:
        response = requests.post(EMBED_URL, json=payload, timeout=60)
        return response.json()["embeddings"]
    except Exception as e:
        print(f"❌ Embedding 失敗: {e}")
        return []

def qwen3_rerank_score(query: str, doc: str) -> float:
    instruction = "根據查詢檢索相關文件"
    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    prompt = f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}{suffix}"
    
    inputs = reranker_tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = reranker_model(**inputs)
        logits = outputs.logits[0, -1, :]
        relevant_logits = torch.tensor([logits[token_false_id], logits[token_true_id]])
        probs = F.softmax(relevant_logits, dim=-1)
        return probs[1].item()

# === 4. 初始化知識庫 ===
def initialize_db():
    print("📡 [步驟 1/2] 初始化 Qdrant 集合...")
    sample_vec = get_embeddings(["check"])[0]
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=len(sample_vec), distance=models.Distance.COSINE)
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="text",
        field_schema=models.TextIndexParams(type="text", tokenizer=models.TokenizerType.MULTILINGUAL)
    )
    file_paths = glob.glob("data_0*.txt")
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    all_points = []
    for path in file_paths:
        file_name = os.path.basename(path)
        with open(path, 'r', encoding='utf-8') as f:
            chunks = splitter.split_text(f.read())
            vectors = get_embeddings(chunks)
            for chunk, vec in zip(chunks, vectors):
                all_points.append(models.PointStruct(
                    id=str(uuid.uuid4()), vector=vec, payload={"text": chunk, "source": file_name}
                ))
    client.upsert(collection_name=COLLECTION_NAME, points=all_points)
    print(f"✅ 匯入完成。")

# === 5. 執行任務 (修復後的 query_points 語法) ===
def run_rag_task():
    df = pd.read_csv("questions.csv")
    df.columns = df.columns.str.strip()
    session_history = {}
    final_answers = []
    final_sources = []

    print("\n🚀 [步驟 2/2] 開始執行 Hybrid Search + Causal Rerank...")

    for index, row in df.iterrows():
        original_q = str(row['題目'])
        cid = "default"
        history_str = session_history.get(cid, "無對話歷史")
        rewritten_q = llm.invoke(f"改寫為搜尋句：{original_q}\n歷史：{history_str}").content.strip()

        # 2. Hybrid Search
        q_vec = get_embeddings([rewritten_q])[0]
        
        # 💡 修正點：使用空查詢但帶有 Filter 的 Prefetch
        # 對於單純的過濾檢索，我們不給 query 具體物件，而是透過 filter 強制篩選
        search_results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                # (1) 向量檢索
                models.Prefetch(query=q_vec, limit=20),
                # (2) 全文檢索修正：不使用 Recommend，改用 Filter 並在 query 中填入空向量
                models.Prefetch(
                    query=[0.0] * len(q_vec), 
                    filter=models.Filter(must=[models.FieldCondition(key="text", match=models.MatchText(text=rewritten_q))]),
                    limit=20
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=15
        ).points

        candidates = [{"text": hit.payload['text'], "source": hit.payload['source']} for hit in search_results]

        # 3. Causal Reranking
        print(f"   [第 {index+1} 題] 正在進行 Rerank (Candidates: {len(candidates)})...")
        # 去重
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c['text'] not in seen:
                unique_candidates.append(c)
                seen.add(c['text'])

        for c in unique_candidates:
            c['score'] = qwen3_rerank_score(rewritten_q, c['text'])
        
        top_3 = sorted(unique_candidates, key=lambda x: x['score'], reverse=True)[:3]
        context_str = "\n".join([f"[{c['source']}]: {c['text']}" for c in top_3]) if top_3 else "無資料"
        
        # 4. 生成
        answer = llm.invoke(f"資訊：\n{context_str}\n問題：{original_q}").content.strip()
        session_history[cid] = history_str + f"\n問：{original_q}\n答：{answer}\n"
        final_answers.append(answer)
        final_sources.append(top_3[0]['source'] if top_3 else "未知")
        print(f"   [回答]: {answer[:20]}...")

    df['標準答案'] = final_answers
    df['來源文件'] = final_sources
    df.to_csv("questions_result_final.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    initialize_db()
    run_rag_task()