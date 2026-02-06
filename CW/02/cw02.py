import os
import io
import pandas as pd
import requests
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

# === 1. 初始化與 VDB 設定 ===
client = QdrantClient(url="http://localhost:6333")

MODES = {
    "COSINE": {"name": "hw_final_cosine", "dist": Distance.COSINE},
    "DOT": {"name": "hw_final_dot", "dist": Distance.DOT},
    "EUCLID": {"name": "hw_final_euclid", "dist": Distance.EUCLID}
}

EMBED_API_URL = "https://ws-04.wade0426.me/embed"

def get_embeddings(texts):
    payload = {"texts": texts, "normalize": True, "batch_size": 32}
    try:
        response = requests.post(EMBED_API_URL, json=payload)
        response.raise_for_status()
        return response.json()['embeddings']
    except Exception as e:
        print(f"❌ Embedding API 錯誤: {e}")
        return []

# === 2. 實作切塊對比印出 (任務 2 & 3) ===

def perform_dual_chunking(file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ 找不到檔案: {file_path}")
        return [], []

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 固定切塊
    fixed_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator="")
    fixed_chunks = [doc.page_content for doc in fixed_splitter.create_documents([text])]

    # 滑動視窗切塊
    sliding_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。 ", "! ", "? ", " ", ""],
        chunk_size=100,
        chunk_overlap=30, 
        add_start_index=True
    )
    sliding_chunks = [doc.page_content for doc in sliding_splitter.create_documents([text])]
    
    print("\n" + "="*20 + " 【2. 固定切塊詳細內容 (Fixed)】 " + "="*20)
    for i, c in enumerate(fixed_chunks):
        # 修正：先處理字串，避免 f-string 反斜線錯誤
        clean_text = c.replace('\n', ' ')
        print(f"Chunk {i+1}: {clean_text}")
        
    print("\n" + "="*20 + " 【3. 滑動視窗詳細內容 (Sliding)】 " + "="*20)
    for i, c in enumerate(sliding_chunks):
        clean_text = c.replace('\n', ' ')
        print(f"Chunk {i+1}: {clean_text}")
    
    return fixed_chunks, sliding_chunks

# === 3. 表格處理過程與結果印出 (任務 6) ===

def process_table_folder(folder_path):
    all_table_data = []
    
    # 優化後的 Prompts
    p1_optimized = "# Role: 商業顧問摘要\n# Task: 識別校區特色與旗艦計畫趨勢...\n# Input:"
    p2_optimized = "# Role: QA 生成助理\n# Task: 生成模擬真實使用者口吻的問答對...\n# Input:"

    print("\n" + "="*20 + " 【表格處理過程與結合結果】 " + "="*20)
    
    if not os.path.exists(folder_path):
        print(f"⚠️ 找不到資料夾: {folder_path}")
        return []

    for file_name in os.listdir(folder_path):
        f_path = os.path.join(folder_path, file_name)
        
        # A. HTML 表格 -> 結合優化後的 Prompt V1 (摘要任務)
        if file_name == "table_html.html":
            with open(f_path, "r", encoding="utf-8") as f:
                html_io = io.StringIO(f.read())
                dfs = pd.read_html(html_io)
                for df in dfs:
                    processed_text = f"{p1_optimized}\n{df.to_string()}"
                    all_table_data.append(processed_text)
                    print(f"\n[處理檔案: {file_name}]\n{processed_text}")

        # B. MD 表格 -> 結合優化後的 Prompt V2 (QA 任務)
        elif file_name == "table_txt.md":
            with open(f_path, "r", encoding="utf-8") as f:
                md_text = f.read()
                processed_text = f"{p2_optimized}\n{md_text}"
                all_table_data.append(processed_text)
                print(f"\n[處理檔案: {file_name}]\n{processed_text}")
            
    return all_table_data

# === 4. 嵌入 VDB (使用 UUID) ===

def upsert_to_vdb(chunks, category):
    if not chunks: return
    vectors = get_embeddings(chunks)
    if not vectors: return
    
    for mode, info in MODES.items():
        if not client.collection_exists(info["name"]):
            client.create_collection(
                collection_name=info["name"],
                vectors_config=VectorParams(size=len(vectors[0]), distance=info["dist"])
            )
        
        # UUID 生成唯一 ID
        points = [
            PointStruct(
                id=uuid.uuid4().hex, 
                vector=vectors[i], 
                payload={"text": chunks[i], "category": category}
            ) for i in range(len(chunks))
        ]
        client.upsert(collection_name=info["name"], points=points)
    print(f"\n✅ {category} 數據已使用 UUID 存入 Qdrant。")

# === 主程式執行 ===

if __name__ == "__main__":
    # 1. 切塊對比內容印出
    _, sliding_chunks = perform_dual_chunking("text.txt")
    
    # 2. 表格結合 Prompt 過程印出
    table_results = process_table_folder("table")
    
    # 3. 儲存
    if sliding_chunks:
        upsert_to_vdb(sliding_chunks, "text_data")
    if table_results:
        upsert_to_vdb(table_results, "table_data")
    
    print("\n🚀 程式執行完畢！所有處理細節已顯示。")