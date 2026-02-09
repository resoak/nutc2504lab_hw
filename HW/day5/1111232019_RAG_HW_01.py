import os
import uuid
import pandas as pd
import requests
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, FilterSelector

# === 修正後的 Import (解決 Splitter 未定義問題) ===
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
    # 增加批量處理 batch_size=32 提高效率
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

# === 2. 檔案處理與切塊 ===

def process_files_and_chunk():
    data_files = [f"data_0{i}.txt" for i in range(1, 6)]
    all_chunks = {"固定大小": [], "滑動視窗": [], "語義切塊": []}
    chunk_source_map = {}
    embeddings_tool = CustomEmbeddings()
    
    # 二次切分器 (當語義塊過大時)
    semantic_sub_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    
    print("\n" + "="*20 + " 1. 開始檔案切塊階段 " + "="*20)
    for file_name in data_files:
        if not os.path.exists(file_name):
            continue
        with open(file_name, "r", encoding="utf-8") as f:
            content = f.read()
        
        print(f"📄 讀取檔案: {file_name} ({len(content)} 字)")
        
        # 1. 固定大小 (CharacterSplitter)
        f_chunks = [d.page_content for d in CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0, separator="").create_documents([content])]
        
        # 2. 滑動視窗 (Recursive)
        s_chunks = [d.page_content for d in RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).create_documents([content])]
        
        # 3. 語義切塊 (優化門檻為 95 percentile)
        sem_splitter = SemanticChunker(
            embeddings_tool, 
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95
        )
        sem_base_docs = sem_splitter.create_documents([content])
        
        sem_chunks_final = []
        for doc in sem_base_docs:
            if len(doc.page_content) > CHUNK_SIZE:
                sub_docs = semantic_sub_splitter.split_text(doc.page_content)
                sem_chunks_final.extend(sub_docs)
            else:
                sem_chunks_final.append(doc.page_content)

        for method, chunks in [("固定大小", f_chunks), ("滑動視窗", s_chunks), ("語義切塊", sem_chunks_final)]:
            all_chunks[method].extend(chunks)
            for c in chunks: 
                chunk_source_map[c] = file_name
        
    return all_chunks, chunk_source_map

# === 3. 向量檢索與評分 ===

def setup_vdb_and_search(all_methods_chunks, chunk_source_map):
    results_for_csv = []
    
    # 讀取問題
    questions_df = pd.read_csv("questions.csv")
    q_texts = questions_df['questions'].astype(str).tolist()
    q_ids = questions_df['q_id'].tolist()
    
    # 固定名稱 Collection 映射
    method_to_coll = {
        "固定大小": "coll_fixed_size",
        "滑動視窗": "coll_sliding_window",
        "語義切塊": "coll_semantic_chunk"
    }
    
    print(f"\n📡 正在批量獲取 {len(q_texts)} 個問題的向量...")
    all_q_vectors = get_embeddings(q_texts)
    
    print("\n" + "="*20 + " 2. 開始批量向量檢索與評分 " + "="*20)

    for method, chunks in all_methods_chunks.items():
        coll_name = method_to_coll[method]
        print(f"\n🛠️ 處理方法: [{method}] | 固定 Collection: {coll_name}")
        
        chunk_vectors = get_embeddings(chunks)
        if not chunk_vectors: continue

        # 檢查 Collection 是否存在，不存在才建
        if not client.collection_exists(coll_name):
            client.create_collection(
                collection_name=coll_name,
                vectors_config=VectorParams(size=len(chunk_vectors[0]), distance=Distance.COSINE)
            )
        else:
            # 🧹 關鍵優化：雖然不刪除 Collection，但清空裡面所有的舊資料 (Point)
            print(f"   🧹 正在清空 {coll_name} 的舊 Point 資料...")
            # 刪除所有滿足 {} 條件的點 (即全刪)
            client.delete(
                collection_name=coll_name,
                points_selector=FilterSelector(filter={})
            )
        
        # Point ID 使用 UUID
        points = [
            PointStruct(id=uuid.uuid4().hex, vector=chunk_vectors[i], payload={"text": chunks[i]}) 
            for i in range(len(chunks))
        ]
        client.upsert(collection_name=coll_name, points=points)

        # 檢索與評分 (limit 改為 3 並合併以增加正確率)
        for i, q_vec in enumerate(all_q_vectors):
            search_res = client.query_points(
                collection_name=coll_name, 
                query=q_vec, 
                limit=3
            ).points
            
            # 合併檢索到的內容
            retrieved_text = "\n".join([h.payload['text'] for h in search_res])
            score = submit_and_get_score(q_ids[i], retrieved_text)
            
            if i % 10 == 0:
                print(f"   📝 Q{q_ids[i]} | Score: {score:.4f}")
            
            results_for_csv.append({
                "q_id": q_ids[i],
                "method": method,
                "retrieve_text": retrieved_text,
                "score": score,
                "source": chunk_source_map.get(retrieved_text.split('\n')[0], "unknown")
            })
            
    return results_for_csv

# === 4. 主程式 ===

if __name__ == "__main__":
    start_time = time.time()
    
    # 1. 執行切塊
    all_chunks, source_map = process_files_and_chunk()
    
    # 2. 執行向量化與評測
    final_results = setup_vdb_and_search(all_chunks, source_map)
    
    # 3. 輸出 CSV
    df_output = pd.DataFrame(final_results)
    df_output.insert(0, 'id', [uuid.uuid4().hex[:8] for _ in range(len(df_output))])
    
    output_name = "1111232019_RAG_HW_01.csv"
    df_output.to_csv(output_name, index=False, encoding="utf-8-sig")
    
    print("\n" + "="*30 + " 3. 執行統計 " + "="*30)
    avg_scores = df_output.groupby('method')['score'].mean()
    for m, s in avg_scores.items():
        print(f"   🔹 {m} 平均分: {s:.4f} | 區塊數: {len(all_chunks[m])}")
    
    print(f"\n✅ 全部完成！總耗時: {time.time() - start_time:.2f} 秒")
    print(f"✅ 結果已儲存至: {output_name}")