import os
import uuid
import pandas as pd
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

# === 0. 配置與初始化 ===
API_KEY = "YOUR_API_KEY" 
EMBED_API_URL = "https://ws-04.wade0426.me/embed"
SUBMIT_URL = "https://hw-01.wade0426.me/submit_answer"

client = QdrantClient(url="http://localhost:6333")

class CustomEmbeddings:
    def embed_documents(self, texts): return get_embeddings(texts)
    def embed_query(self, text): return get_embeddings([text])[0]

# === 1. 功能函數 ===

def get_embeddings(texts):
    payload = {"texts": texts, "normalize": True, "batch_size": 32}
    try:
        response = requests.post(EMBED_API_URL, json=payload)
        response.raise_for_status()
        return response.json()['embeddings']
    except Exception as e:
        print(f"❌ Embedding API 錯誤: {e}")
        return []

def submit_and_get_score(q_id, answer):
    payload = {"q_id": q_id, "student_answer": answer}
    try:
        response = requests.post(SUBMIT_URL, json=payload)
        return response.json().get("score", 0) if response.status_code == 200 else 0
    except:
        return 0

# === 2. 檔案處理與三種切塊 ===

def process_files_and_chunk():
    data_files = [f"data_0{i}.txt" for i in range(1, 6)]
    all_chunks = {"固定大小": [], "滑動視窗": [], "語義切塊": []}
    chunk_source_map = {}
    embeddings_tool = CustomEmbeddings()
    
    print("\n" + "="*20 + " 1. 開始檔案切塊階段 " + "="*20)
    for file_name in data_files:
        if not os.path.exists(file_name):
            print(f"⚠️ 跳過不存在的檔案: {file_name}")
            continue
        with open(file_name, "r", encoding="utf-8") as f:
            content = f.read()
        
        print(f"📄 讀取檔案: {file_name} ({len(content)} 字)")
        
        f_chunks = [d.page_content for d in CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator="").create_documents([content])]
        s_chunks = [d.page_content for d in RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).create_documents([content])]
        sem_chunks = [d.page_content for d in SemanticChunker(embeddings_tool).create_documents([content])]

        for method, chunks in [("固定大小", f_chunks), ("滑動視窗", s_chunks), ("語義切塊", sem_chunks)]:
            all_chunks[method].extend(chunks)
            for c in chunks: chunk_source_map[c] = file_name
        
    return all_chunks, chunk_source_map

# === 3. 向量檢索與評分 (優化警告部分) ===

# === 3. 向量檢索與評分 (優化並新增 Collection 名稱顯示) ===

def setup_vdb_and_search(all_methods_chunks, chunk_source_map):
    results_for_csv = []
    
    # 讀取問題並一次性進行批量 Embedding
    questions_df = pd.read_csv("questions.csv")
    q_texts = questions_df['questions'].astype(str).tolist()
    q_ids = questions_df['q_id'].tolist()
    
    print(f"\n📡 正在批量獲取 {len(q_texts)} 個問題的向量...")
    all_q_vectors = get_embeddings(q_texts)
    
    print("\n" + "="*20 + " 2. 開始批量向量檢索與評分 " + "="*20)

    for method, chunks in all_methods_chunks.items():
        coll_name = f"hw_{uuid.uuid4().hex[:8]}"
        print(f"\n🛠️ 處理方法: [{method}] | Collection: {coll_name}")
        
        # 🚀 批量 1: 一次性獲取所有 Chunks 的向量
        print(f"   ⬆️ 正在上傳 {len(chunks)} 個文本區塊...")
        chunk_vectors = get_embeddings(chunks)
        
        if client.collection_exists(coll_name):
            client.delete_collection(coll_name)
        
        client.create_collection(
            collection_name=coll_name,
            vectors_config=VectorParams(size=len(chunk_vectors[0]), distance=Distance.COSINE)
        )
        
        points = [
            PointStruct(id=i, vector=chunk_vectors[i], payload={"text": chunks[i]}) 
            for i in range(len(chunks))
        ]
        client.upsert(collection_name=coll_name, points=points)

        # 🚀 批量 2: 檢索與評分優化
        # 雖然評分 API 通常是單點提交，但我們可以優化檢索邏輯
        for i, q_vec in enumerate(all_q_vectors):
            # 這裡可以使用 Qdrant 的 batch 搜尋 API，但為了維持 logic 清晰，我們批量處理變數
            search_res = client.query_points(
                collection_name=coll_name, 
                query=q_vec, 
                limit=1
            ).points
            
            retrieved_text = search_res[0].payload['text'] if search_res else ""
            
            # 提交評分 (此處若 API 支援 Batch 提交會更快)
            score = submit_and_get_score(q_ids[i], retrieved_text)
            
            if i % 5 == 0: # 減少 log 刷屏，每 5 題印一次
                print(f"   📝 已處理 Q{q_ids[i]} | Score: {score:.4f}")
            
            results_for_csv.append({
                "q_id": q_ids[i],
                "method": method,
                "retrieve_text": retrieved_text,
                "score": score,
                "source": chunk_source_map.get(retrieved_text, "unknown")
            })
        
        # 選項：清理 Collection 節省記憶體
        # client.delete_collection(coll_name)
            
    return results_for_csv

# === 4. 主程式 ===

if __name__ == "__main__":
    all_chunks, source_map = process_files_and_chunk()
    final_results = setup_vdb_and_search(all_chunks, source_map)
    
    df_output = pd.DataFrame(final_results)
    df_output.insert(0, 'id', [uuid.uuid4().hex[:8] for _ in range(len(df_output))])
    
    output_name = "1111232019_RAG_HW_01.csv"
    df_output.to_csv(output_name, index=False, encoding="utf-8-sig")
    
    print("\n" + "="*30 + " 3. 最終 CSV 執行結果 (60 筆) " + "="*30)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', 60)
    # 打印前 60 筆的重要欄位供快速檢查
    print(df_output[['id', 'q_id', 'method', 'score', 'source']])
    
    print("\n" + "="*60)
    avg_scores = df_output.groupby('method')['score'].mean()
    print("💡 各切塊方法平均分數統計：")
    for m, s in avg_scores.items():
        print(f"   🔹 {m}: {s:.4f}")
    print("="*60)