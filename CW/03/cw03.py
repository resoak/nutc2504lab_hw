import os
import glob
import pandas as pd
import uuid
import requests
import sys
import time
from typing import List

# LangChain 與模型相關組件
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

# === 1. 配置與初始化 ===
VLM_BASE_URL = "https://ws-02.wade0426.me/v1"
VLM_MODEL = "google/gemma-3-27b-it"
EMBED_URL = "https://ws-04.wade0426.me/embed"
COLLECTION_NAME = "gemma_multi_turn_rag"

# 請確保 API Key 正確
llm = ChatOpenAI(
    base_url=VLM_BASE_URL,
    api_key="YOUR_API_KEY", 
    model=VLM_MODEL,
    temperature=0,
    timeout=60 
)

client = QdrantClient(url="http://localhost:6333")

# === 2. 高速向量化工具函數 (支援批次處理與重試) ===
def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    if not texts: return []
    payload = {"texts": texts, "normalize": True, "task_description": "檢索技術與生活文件"}
    for attempt in range(3):
        try:
            response = requests.post(EMBED_URL, json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get("embeddings", [])
        except Exception as e:
            print(f"  ⚠️ Embedding 嘗試 {attempt+1} 失敗: {e}")
            time.sleep(2)
    return []

# === 3. 初始化知識庫 (高速版) ===
def initialize_db():
    print("\n" + "="*50)
    print("📡 [步驟 1/2] 正在高速初始化知識庫...")
    
    sample = get_embeddings_batch(["check"])
    if not sample:
        print("🛑 向量伺服器連線失敗，程式停止。")
        sys.exit(1)
        
    dim = len(sample[0])
    # 快速重置 Collection
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
    )
    
    file_paths = glob.glob("data_0*.txt")
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    for path in file_paths:
        file_name = os.path.basename(path)
        print(f"📖 處理檔案: {file_name}...", end="", flush=True)
        with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
            content = f.read().replace('\ufffd', '')
            chunks = splitter.split_text(content)
            vectors = get_embeddings_batch(chunks)
            if vectors:
                points = [models.PointStruct(
                    id=str(uuid.uuid4()), 
                    vector=v, 
                    payload={"text": c, "source": file_name}
                ) for c, v in zip(chunks, vectors)]
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                print(f" ✅ ({len(chunks)} 區塊)")
            else:
                print(" ❌ 向量化失敗")

# === 4. 執行 RAG 任務 (修正型別衝突與 502 錯誤) ===
def run_rag_task():
    print("\n" + "="*50)
    input_file = "Re_Write_questions.csv" 
    prompt_file = "Prompt_ReWrite.txt"
    output_file = "Re_Write_questions_result.csv"

    if not os.path.exists(input_file) or not os.path.exists(prompt_file):
        print("❌ 錯誤：找不到必要檔案。")
        return

    with open(prompt_file, "r", encoding="utf-8") as f:
        rewrite_instruction = f.read()

    # 讀取 CSV
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    
    # --- 關鍵修正：預先強制轉換型別為字串物件，避免 LossySetitemError ---
    df['answer'] = ""
    df['answer'] = df['answer'].astype(object)
    df['source'] = ""
    df['source'] = df['source'].astype(object)
    # -------------------------------------------------------------

    session_history = {} 
    print(f"🚀 [步驟 2/2] 開始處理問題集 (共 {len(df)} 題)...")

    for index, row in df.iterrows():
        cid = str(row['conversation_id'])
        original_q = str(row['questions']) 
        history = session_history.get(cid, "")

        print(f"\n--- [正在處理第 {index+1} 題] (CID: {cid}) ---")

        # A. 問題改寫 (含簡單重試)
        rewritten_q = original_q
        for _ in range(2):
            try:
                rewrite_prompt = f"{rewrite_instruction}\n\n[歷史]:\n{history}\n\n[問題]:\n{original_q}\n\n搜尋句："
                rewritten_q = llm.invoke(rewrite_prompt).content.strip()
                print(f"🔍 搜尋句: {rewritten_q}")
                break
            except: time.sleep(2)

        # B. 檢索
        q_vec_list = get_embeddings_batch([rewritten_q])
        context, top_source = "", "未知"
        if q_vec_list:
            q_vec = q_vec_list[0]
            hits = client.query_points(collection_name=COLLECTION_NAME, query=q_vec, limit=3).points
            context = "\n".join([h.payload['text'] for h in hits])
            top_source = hits[0].payload['source'] if hits else "未知來源"
            for i, hit in enumerate(hits):
                print(f"  📍 匹配項 {i+1}: {hit.payload['text'][:30]}...")

        # C. 回答生成 (處理 502 Bad Gateway)
        final_prompt = (
            f"你是一個助手，請根據資訊回答問題。請使用正確的繁體中文，避免錯字。\n"
            f"若資訊中出現編碼偏移（如『虨擬』、『斧理器』），請自動修正為正確名詞（如『虛擬』、『處理器』）。\n\n"
            f"【資訊】：\n{context}\n\n"
            f"【問題】：{rewritten_q}\n回答："
        )
        
        answer = "伺服器暫時連線失敗，請檢查後端狀態。"
        for attempt in range(3):
            try:
                answer_content = llm.invoke(final_prompt).content.strip().replace('\ufffd', '')
                answer = answer_content
                print(f"✨ AI 回答成功")
                break
            except Exception as e:
                print(f"  ⚠️ 生成失敗 (嘗試 {attempt+1})，原因: {e}")
                time.sleep(7) # 遇到 502/504 時，讓伺服器喘息一下

        # 填入結果
        df.at[index, 'answer'] = answer
        df.at[index, 'source'] = top_source
        
        # 更新對話歷史
        session_history[cid] = history + f"問：{original_q}\n答：{answer}\n"

    # 輸出最終結果
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n" + "="*50)
    print(f"🎉 任務處理完畢！結果儲存至: {output_file}")

if __name__ == "__main__":
    initialize_db()
    run_rag_task()