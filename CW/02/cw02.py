import os
import io
import pandas as pd
import requests
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import ChatOpenAI

# === 0. 初始化 LLM ===
llm = ChatOpenAI(
    base_url="https://ws-05.huannago.com/v1",
    api_key="YOUR_API_KEY", # ⚠️ 請在此填入您的 API Key
    model="google/gemma-3-27b-it",
    temperature=0.7
)

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

# === 2. 實作文字切塊對比印出 (text.txt) ===

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
    
    print("\n" + "="*20 + " 【2. text.txt 固定切塊 (Fixed)】 " + "="*20)
    for i, c in enumerate(fixed_chunks):
        clean_text = c.replace('\n', ' ')
        print(f"Chunk {i+1}: {clean_text}")
        
    print("\n" + "="*20 + " 【3. text.txt 滑動視窗 (Sliding)】 " + "="*20)
    for i, c in enumerate(sliding_chunks):
        clean_text = c.replace('\n', ' ')
        print(f"Chunk {i+1}: {clean_text}")
    
    return fixed_chunks, sliding_chunks

# === 3. 表格處理：LLM 轉換與生成後切塊 ===

def process_table_via_llm_and_chunk(folder_path):
    """讀取表格，交給 LLM 生成文字資訊，再進行切塊"""
    # 讀取本地 Prompt 檔案
    p1_path = os.path.join(folder_path, "Prompt_table_v1.txt")
    p2_path = os.path.join(folder_path, "Prompt_table_v2.txt")
    p1_prompt = open(p1_path, "r", encoding="utf-8").read() if os.path.exists(p1_path) else "請摘要此表格"
    p2_prompt = open(p2_path, "r", encoding="utf-8").read() if os.path.exists(p2_path) else "請根據此表格生成問答"

    all_llm_chunks = []
    
    # 修正處：正確的名稱為 RecursiveCharacterTextSplitter
    table_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150, 
        chunk_overlap=20, 
        separators=["\n\n", "\n", "。", " "]
    )

    print("\n" + "="*20 + " 【LLM 表格處理與切塊過程】 " + "="*20)
    
    if not os.path.exists(folder_path):
        print(f"⚠️ 找不到資料夾: {folder_path}")
        return []

    for file_name in os.listdir(folder_path):
        f_path = os.path.join(folder_path, file_name)
        llm_response_text = ""

        # A. HTML -> LLM 摘要 (V1)
        if file_name == "table_html.html":
            with open(f_path, "r", encoding="utf-8") as f:
                html_io = io.StringIO(f.read())
                dfs = pd.read_html(html_io)
                for df in dfs:
                    input_content = f"{p1_prompt}\n表格數據：\n{df.to_string()}"
                    print(f"正在請求 LLM 生成 {file_name} 的摘要報告...")
                    response = llm.invoke(input_content)
                    llm_response_text = response.content

        # B. Markdown -> LLM QA (V2)
        elif file_name == "table_txt.md":
            with open(f_path, "r", encoding="utf-8") as f:
                md_text = f.read()
                input_content = f"{p2_prompt}\n表格數據：\n{md_text}"
                print(f"正在請求 LLM 生成 {file_name} 的問答數據...")
                response = llm.invoke(input_content)
                llm_response_text = response.content
        
        # 處理 LLM 產出的文字並切塊
        if llm_response_text:
            print(f"\n--- LLM 生成內容 ({file_name}) ---\n{llm_response_text}\n")
            
            # 對 LLM 的長回答進行切塊，以便更好的檢索
            chunks = [doc.page_content for doc in table_text_splitter.create_documents([llm_response_text])]
            
            print(f"--- LLM 內容切塊結果 ({file_name}) ---")
            for i, chunk in enumerate(chunks):
                clean_chunk = chunk.replace('\n', ' ')
                print(f"LLM_Chunk {i+1}: {clean_chunk}")
                all_llm_chunks.append(chunk)
            
    return all_llm_chunks

# === 4. 嵌入 VDB (UUID) ===

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
        # 使用 UUID
        points = [
            PointStruct(id=uuid.uuid4().hex, vector=vectors[i], payload={"text": chunks[i], "category": category}) 
            for i in range(len(chunks))
        ]
        client.upsert(collection_name=info["name"], points=points)
    print(f"\n✅ {category} 數據已成功存入 Qdrant。")

# === 主程式 ===

if __name__ == "__main__":
    # 1. 處理原始文字
    _, sliding_text = perform_dual_chunking("text.txt")
    
    # 2. 透過 LLM 處理表格並切塊
    llm_chunks = process_table_via_llm_and_chunk("table")
    
    # 3. 儲存至資料庫
    if sliding_text:
        upsert_to_vdb(sliding_text, "text_data")
    if llm_chunks:
        upsert_to_vdb(llm_chunks, "llm_enhanced_table_data")
    
    print("\n🚀 任務完成！LLM 生成的內容已成功切塊並儲存。")