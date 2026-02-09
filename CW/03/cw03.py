import os
import glob
import pandas as pd
import uuid
import time
from typing import List, Dict

# LangChain 相關組件
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests

# Qdrant 相關組件
from qdrant_client import QdrantClient, models

# === 1. 配置與初始化 ===
VLM_BASE_URL = "https://ws-05.huannago.com/v1"
VLM_MODEL = "google/gemma-3-27b-it"
EMBED_URL = "https://ws-04.wade0426.me/embed"
COLLECTION_NAME = "gemma_multi_turn_rag"

# 初始化 LLM (Gemma-3-27b-it)
llm = ChatOpenAI(
    base_url=VLM_BASE_URL,
    api_key="YOUR_API_KEY", # ⚠️ 請在此處填入您的 API Key
    model=VLM_MODEL,
    temperature=0,
    timeout=120
)

# 連線至本地 Qdrant (Dashboard: http://localhost:6333)
client = QdrantClient(url="http://localhost:6333")

# === 2. 向量化工具函數 ===
def get_embeddings(texts: List[str]) -> List[List[float]]:
    payload = {"texts": texts, "normalize": True, "task_description": "檢索技術與生活文件"}
    try:
        response = requests.post(EMBED_URL, json=payload, timeout=60)
        return response.json()["embeddings"]
    except Exception as e:
        print(f"❌ Embedding 失敗: {e}")
        return []

# === 3. 初始化知識庫 ===
def initialize_db():
    print("\n" + "="*50)
    print("📡 [步驟 1/2] 正在初始化本地 Qdrant 知識庫...")
    print("="*50)
    
    sample_vec = get_embeddings(["check"])[0]
    dim = len(sample_vec)
    
    if client.collection_exists(COLLECTION_NAME):
        print(f"🗑️  偵測到舊集合，正在刪除並重建: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
    )
    
    file_paths = glob.glob("data_0*.txt")
    splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
    all_points = []
    
    for path in file_paths:
        file_name = os.path.basename(path)
        print(f"📖 正在讀取並切分檔案: {file_name}")
        with open(path, 'r', encoding='utf-8') as f:
            chunks = splitter.split_text(f.read())
            vectors = get_embeddings(chunks)
            for chunk, vec in zip(chunks, vectors):
                all_points.append(models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={"text": chunk, "source": file_name}
                ))
    
    client.upsert(collection_name=COLLECTION_NAME, points=all_points)
    print(f"✅ 知識庫匯入完成，共計 {len(all_points)} 個資料點。")

# === 4. 執行多輪 RAG 任務 (全過程 Print) ===
def run_rag_task():
    input_file = "Re_Write_questions.csv" 
    if not os.path.exists(input_file):
        print(f"❌ 找不到來源檔案: {input_file}")
        return
    
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()
    
    if os.path.exists("Prompt_ReWrite.txt"):
        with open("Prompt_ReWrite.txt", "r", encoding="utf-8") as f:
            rewrite_instruction = f.read()
    else:
        rewrite_instruction = "你是一個查詢重寫專家。請根據對話歷史將最新問題改寫為獨立的搜尋語句。"

    session_history = {} 
    final_answers = []
    final_sources = []

    print("\n" + "="*50)
    print(f"🚀 [步驟 2/2] 開始處理問題集: {input_file}")
    print("="*50)

    for index, row in df.iterrows():
        cid = str(row['conversation_id'])
        original_q = str(row['questions']) 
        history_str = session_history.get(cid, "無對話歷史")

        print(f"\n👉 [第 {index+1} 題] 會話 ID: {cid}")
        print(f"   [原始問題]: {original_q}")

        # Step 1: Query Rewrite
        print(f"   [正在改寫查詢中...]")
        rewrite_prompt = f"{rewrite_instruction}\n\n[對話歷史]:\n{history_str}\n\n[最新問題]:\n{original_q}\n\n請直接輸出重寫後的搜尋語句："
        try:
            rewritten_q = llm.invoke(rewrite_prompt).content.strip()
            print(f"   [改寫結果]: {rewritten_q}")
        except Exception as e:
            print(f"   ⚠️ 改寫失敗 ({e})，使用原句搜尋。")
            rewritten_q = original_q

        # Step 2: Retrieval
        print(f"   [正在檢索向量資料庫...]")
        q_vec = get_embeddings([rewritten_q])[0]
        search_results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_vec,
            limit=5
        ).points
        
        context_list = [hit.payload['text'] for hit in search_results]
        context_str = "\n".join(context_list)
        top_source = search_results[0].payload['source'] if search_results else "未知來源"
        
        print(f"   [檢索到來源]: {top_source}")
        # print(f"   [參考片段]: {context_list[0][:50]}...") # 若想看更細可解鎖這行

        # Step 3: Generation
        print(f"   [正在生成最終回答...]")
        final_prompt = f"""請嚴格根據參考資訊回答。資訊不足請回「抱歉，我無法回答」。
【參考資訊】：
{context_str}
【問題】：{rewritten_q}
回答："""
        
        try:
            answer = llm.invoke(final_prompt).content.strip()
            print(f"   [機器回答]: {answer[:50]}...")
        except Exception as e:
            answer = "抱歉，系統生成回答時出錯。"
            print(f"   ❌ 回答生成失敗: {e}")
        
        # 更新歷史
        session_history[cid] = history_str + f"\n問：{original_q}\n答：{answer}\n"
        
        final_answers.append(answer)
        final_sources.append(top_source)
        time.sleep(0.5)

    # 儲存結果
    df['answer'] = final_answers
    df['source'] = final_sources
    
    output_csv = "Re_Write_questions_result.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print("\n" + "="*50)
    print(f"🎉 任務圓滿完成！")
    print(f"💾 結果檔案已儲存至: {output_csv}")
    print("="*50)

if __name__ == "__main__":
    initialize_db()
    run_rag_task()