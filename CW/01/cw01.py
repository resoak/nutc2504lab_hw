import requests
import time
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue
)
# 引入 LangChain 切塊工具
from langchain_text_splitters import CharacterTextSplitter

# === 1. 初始化與設定 ===
client = QdrantClient(url="http://localhost:6333")

# 定義三種模式，用於對比不同計算法的分數
MODES = {
    "COSINE": {"name": "hw_final_cosine", "dist": Distance.COSINE},
    "DOT": {"name": "hw_final_dot", "dist": Distance.DOT},
    "EUCLID": {"name": "hw_final_euclid", "dist": Distance.EUCLID}
}

# === 2. 核心功能函數 ===

def get_embeddings(texts):
    """從 API 獲取向量"""
    url = "https://ws-04.wade0426.me/embed"
    payload = {"texts": texts, "normalize": True, "batch_size": 32}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()['embeddings']
    except Exception as e:
        print(f"❌ Embedding Error: {e}")
        return []

def split_text_into_chunks(text):
    """實作固定切塊優化：確保產生 5 個以上的分塊"""
    # 透過 chunk_size 控制切塊長度，確保資料點數量符合作業要求
    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=35, 
        chunk_overlap=5,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    print(f"✂️ 文本切割完成，共產生 {len(chunks)} 個 Point。")
    return chunks

def initialize_all_collections():
    """動態偵測維度並建立 Collection (解決 size 寫死的問題)"""
    print("🔍 偵測模型維度中...")
    test_vec = get_embeddings(["check dimension"])
    if not test_vec: return
    
    # 動態取得維度，不再寫死數字
    dynamic_size = len(test_vec[0])
    print(f"📏 偵測到維度: {dynamic_size}\n")

    for mode, info in MODES.items():
        # 新版 API 寫法：避免 DeprecationWarning
        if client.collection_exists(info["name"]):
            client.delete_collection(info["name"])
        
        client.create_collection(
            collection_name=info["name"],
            vectors_config=VectorParams(size=dynamic_size, distance=info["dist"]),
        )
        # 建立 Payload 索引加速分類搜尋
        client.create_payload_index(info["name"], "category", "keyword")
        print(f"🚀 已初始化 [{mode}] Collection")

def upsert_data_to_all(chunks, category):
    """批次將分塊資料轉成向量並上傳"""
    print(f"\n📦 正在執行批次 Embedding 並上傳到三種資料庫...")
    vectors = get_embeddings(chunks)
    if not vectors: return

    for mode, info in MODES.items():
        points = [
            PointStruct(
                id=int(time.time() * 1000) + i, 
                vector=vectors[i],
                payload={"text": chunks[i], "category": category}
            ) for i in range(len(chunks))
        ]
        client.upsert(collection_name=info["name"], points=points)
        print(f"✅ [{mode}] 庫資料同步完成")

# === 3. 召回與對比搜尋 ===
def search_comparison(query_text, category=None):
    """對比三種計算法在同樣搜尋條件下的召回結果"""
    print(f"\n" + "="*65)
    print(f"🔎 搜尋字句: 「{query_text}」 | 指定分類: {category or '全部'}")
    print("="*65)

    query_vec = get_embeddings([query_text])[0]
    
    # 設定分類篩選器
    search_filter = None
    if category:
        search_filter = Filter(must=[FieldCondition(key="category", match=MatchValue(value=category))])

    for mode, info in MODES.items():
        results = client.query_points(
            collection_name=info["name"],
            query=query_vec,
            query_filter=search_filter,
            limit=3
        )
        print(f"\n🔹 模式: {mode} (分數越高越接近)")
        if not results.points:
            print("   ⚠️ 找不到符合條件的結果")
        for p in results.points:
            print(f"   [Score: {p.score:8.4f}] -> {p.payload['text'].strip()}")

# === 4. 執行流程 ===
if __name__ == "__main__":
    # 步驟 1: 初始化三種計算法的 VDB
    initialize_all_collections()

    # 步驟 2: 準備一段長文本以供切塊 (確保 Point 數量 >= 5)
    long_text = """
    機器學習是人工智慧的一個子領域，專注於演算法的開發。
    深度學習利用神經網路結構來模擬人類大腦的學習方式。
    大型語言模型如 GPT-4 具備強大的文本理解與生成能力。
    向量資料庫 Qdrant 提供了高效的相似度檢索功能。
    語意搜尋能理解詞句背後的真實意義，而非僅靠關鍵字。
    在 RAG 架構中，切塊技術與向量化是檢索精準度的關鍵。
    """

    # 步驟 3: 切塊並上傳
    chunks = split_text_into_chunks(long_text)
    upsert_data_to_all(chunks, category="ai_tech")

    # 步驟 4: 召回測試
    search_comparison("什麼是人工智慧相關技術？", category="ai_tech")