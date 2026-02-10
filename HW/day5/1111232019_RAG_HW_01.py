import os
import uuid
import pandas as pd
import requests
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# === 修正後的 Import ===
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

# === 0. 配置與初始化 ===
API_KEY = "YOUR_API_KEY" 
EMBED_API_URL = "https://ws-04.wade0426.me/embed"
SUBMIT_URL = "https://hw-01.wade0426.me/submit_answer"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

client = QdrantClient(url="http://localhost:6333")

class CustomEmbeddings:
    def embed_documents(self, texts): return get_embeddings(texts)
    def embed_query(self, text): return get_embeddings([text])[0]

# === 1. 功能函數 ===

def get_embeddings(texts):
    if not texts: return []
    payload = {"texts": texts, "normalize": True, "batch_size": 32}
    try:
        response = requests.post(EMBED_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()['embeddings']
    except Exception as e:
        print(f"❌ Embedding API 錯誤: {e}")
        return []

def submit_and_get_score(q_id, answer):
    payload = {"q_id": q_id, "student_answer": answer}
    try:
        response = requests.post(SUBMIT_URL, json=payload, timeout=20)
        return response.json().get("score", 0) if response.status_code == 200 else 0
    except:
        return 0

# === 2. 檔案處理與切塊 (加入 Metadata) ===

def process_files_and_chunk():
    data_files = [f"data_0{i}.txt" for i in range(1, 6)]
    all_chunks_data = {"固定大小": [], "滑動視窗": [], "語義切塊": []}
    embeddings_tool = CustomEmbeddings()
    
    semantic_sub_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    
    print("\n" + "="*20 + " 1. 開始檔案切塊階段 " + "="*20)
    for file_name in data_files:
        if not os.path.exists(file_name):
            continue
        with open(file_name, "r", encoding="utf-8") as f:
            content = f.read()
        
        print(f"📄 讀取檔案: {file_name} ({len(content)} 字)")
        
        # 1. 固定大小
        f_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0, separator="")
        for c in [d.page_content for d in f_splitter.create_documents([content])]:
            all_chunks_data["固定大小"].append({"text": c, "source": file_name})
        
        # 2. 滑動視窗
        s_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        for c in [d.page_content for d in s_splitter.create_documents([content])]:
            all_chunks_data["滑動視窗"].append({"text": c, "source": file_name})
        
        # 3. 語義切塊
        sem_splitter = SemanticChunker(
            embeddings_tool, 
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95
        )
        sem_base_docs = sem_splitter.create_documents([content])
        
        for doc in sem_base_docs:
            if len(doc.page_content) > CHUNK_SIZE:
                sub_docs = semantic_sub_splitter.split_text(doc.page_content)
                for sub_c in sub_docs:
                    all_chunks_data["語義切塊"].append({"text": sub_c, "source": file_name})
            else:
                all_chunks_data["語義切塊"].append({"text": doc.page_content, "source": file_name})
        
    return all_chunks_data

# === 3. 向量檢索與評分 (修改後：檢查資料庫是否存在) ===

def setup_vdb_and_search():
    results_for_csv = []
    
    # 讀取問題
    questions_df = pd.read_csv("questions.csv")
    q_texts = questions_df['questions'].astype(str).tolist()
    q_ids = questions_df['q_id'].tolist()
    
    method_to_coll = {
        "固定大小": "coll_fixed_size",
        "滑動視窗": "coll_sliding_window",
        "語義切塊": "coll_semantic_chunk"
    }
    
    print(f"\n📡 正在批量獲取 {len(q_texts)} 個問題的向量...")
    all_q_vectors = get_embeddings(q_texts)
    
    # 延遲切塊：只有在必要時才呼叫切塊函數
    all_chunks_data = None 

    print("\n" + "="*20 + " 2. 向量檢索與評分階段 " + "="*20)

    for method, coll_name in method_to_coll.items():
        print(f"\n🛠️ 正在確認方法: [{method}]")
        
        # --- 修改處：檢查 Collection 是否存在 ---
        if not client.collection_exists(collection_name=coll_name):
            print(f"ℹ️ {coll_name} 不存在，開始初始化資料...")
            
            # 只有第一次需要時才執行切塊
            if all_chunks_data is None:
                all_chunks_data = process_files_and_chunk()
            
            chunk_items = all_chunks_data[method]
            texts = [item['text'] for item in chunk_items]
            sources = [item['source'] for item in chunk_items]
            
            chunk_vectors = get_embeddings(texts)
            if not chunk_vectors:
                print(f"⚠️ 無法取得向量，跳過方法: {method}")
                continue

            # 建立 Collection
            client.create_collection(
                collection_name=coll_name,
                vectors_config=VectorParams(size=len(chunk_vectors[0]), distance=Distance.COSINE)
            )
            
            # 存入資料
            points = [
                PointStruct(
                    id=uuid.uuid4().hex, 
                    vector=chunk_vectors[i], 
                    payload={"text": texts[i], "source": sources[i]}
                ) for i in range(len(texts))
            ]
            client.upsert(collection_name=coll_name, points=points)
            print(f"✅ {coll_name} 資料初始化完成。")
        else:
            print(f"✅ {coll_name} 已存在，跳過上傳步驟。")

        # --- 執行檢索與評分 (無論是否為新建立都會執行) ---
        for i, q_vec in enumerate(all_q_vectors):
            search_res = client.query_points(
                collection_name=coll_name, 
                query=q_vec, 
                limit=3
            ).points
            
            retrieved_content = "\n".join([h.payload['text'] for h in search_res])
            unique_sources = list(set([h.payload['source'] for h in search_res]))
            source_str = ",".join(unique_sources)
            
            score = submit_and_get_score(q_ids[i], retrieved_content)
            
            if i % 20 == 0:
                print(f"   📝 Q{q_ids[i]} | Score: {score:.4f} | Method: {method}")
            
            results_for_csv.append({
                "q_id": q_ids[i],
                "method": method,
                "retrieve_text": retrieved_content,
                "score": score,
                "source": source_str
            })
            
    return results_for_csv

# === 4. 主程式 ===

if __name__ == "__main__":
    start_time = time.time()
    
    # 修改：直接進入資料庫處理流程，內含切塊邏輯
    final_results = setup_vdb_and_search()
    
    # 輸出 CSV
    df_output = pd.DataFrame(final_results)
    df_output.insert(0, 'id', [uuid.uuid4().hex[:8] for _ in range(len(df_output))])
    
    output_name = "1111232019_RAG_HW_01.csv"
    df_output.to_csv(output_name, index=False, encoding="utf-8-sig")
    
    print("\n" + "="*30 + " 3. 執行統計 " + "="*30)
    if not df_output.empty:
        avg_scores = df_output.groupby('method')['score'].mean()
        for m, s in avg_scores.items():
            print(f"   🔹 {m} 平均分: {s:.4f}")
    
    print(f"\n✅ 全部完成！總耗時: {time.time() - start_time:.2f} 秒")
    print(f"✅ 結果已儲存至: {output_name}")