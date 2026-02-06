import requests
import time
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue
)

# === 1. 初始化與設定 ===
client = QdrantClient(url="http://localhost:6333")

# 定義三種計算法模式
MODES = {
    "COSINE": {"name": "hw_cosine_final", "dist": Distance.COSINE},
    "DOT": {"name": "hw_dot_final", "dist": Distance.DOT},
    "EUCLID": {"name": "hw_euclid_final", "dist": Distance.EUCLID}
}

# === 2. Embedding 核心函數 (封裝成函數) ===
def get_embeddings(texts):
    """將模型調用封裝，動態獲取向量"""
    url = "https://ws-04.wade0426.me/embed"
    payload = {
        "texts": texts,
        "normalize": True,
        "batch_size": 32
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()['embeddings']
    except Exception as e:
        print(f"❌ Embedding Error: {e}")
        return []

# === 3. 初始化 VDB (自動偵測 Size & 官方推薦新寫法) ===
def initialize_all_vdbs():
    """先測量維度，再根據不同計算法建立 Collection"""
    print("🔍 正在透過 API 偵測模型向量維度...")
    
    # 【解決關鍵】先拿一筆資料測試維度，不把 size 寫死
    test_vec = get_embeddings(["dimension check"])
    if not test_vec or len(test_vec) == 0:
        print("❌ 無法取得向量，請檢查 API 狀態。")
        return
    
    dynamic_size = len(test_vec[0])
    print(f"📏 偵測到模型維度為: {dynamic_size}\n")

    for mode, info in MODES.items():
        col_name = info["name"]
        
        # 修正 DeprecationWarning: 檢查是否存在 -> 刪除 -> 建立
        if client.collection_exists(collection_name=col_name):
            client.delete_collection(collection_name=col_name)
            print(f"🗑️ 已清理舊的 [{mode}] 集合")
        
        # 建立 Collection，將偵測到的 dynamic_size 傳入
        client.create_collection(
            collection_name=col_name,
            vectors_config=VectorParams(
                size=dynamic_size, 
                distance=info["dist"]
            ),
        )
        
        # 建立分類索引 (加速分類搜尋)
        client.create_payload_index(col_name, "category", "keyword")
        print(f"🚀 已初始化 [{mode}] 資料庫: {col_name}")

# === 4. 批次上傳函數 (Batch Upsert) ===
def batch_upsert_to_all(data_list):
    """將同一份資料同步批次上傳到三個 Collection"""
    print(f"\n📦 正在進行批次處理 (共 {len(data_list)} 筆資料)...")
    texts = [item["text"] for item in data_list]
    vectors = get_embeddings(texts)
    
    if not vectors: return

    for mode, info in MODES.items():
        points = [
            PointStruct(
                id=int(time.time() * 1000) + i, 
                vector=vectors[i],
                payload={
                    "text": data_list[i]["text"],
                    "category": data_list[i]["category"]
                }
            ) for i in range(len(data_list))
        ]
        client.upsert(collection_name=info["name"], points=points)
        print(f"✅ 資料已批次同步至 [{mode}] 庫")

# === 5. 對比搜尋 (支援分類篩選) ===
def compare_search(query_text, target_category=None):
    """一次對比三種算法的搜尋結果，並過濾分類"""
    print(f"\n" + "="*60)
    print(f"🔎 搜尋對比: 「{query_text}」 | 分類過濾: {target_category or '全部'}")
    print("="*60)

    query_vector = get_embeddings([query_text])[0]
    
    # 建立 Qdrant 分類篩選器
    search_filter = None
    if target_category:
        search_filter = Filter(
            must=[FieldCondition(key="category", match=MatchValue(value=target_category))]
        )

    for mode, info in MODES.items():
        results = client.query_points(
            collection_name=info["name"],
            query=query_vector,
            query_filter=search_filter,
            limit=2
        )
        
        print(f"\n🔹 模式: {mode}")
        if not results.points:
            print("   ⚠️ 無匹配結果")
        for p in results.points:
            print(f"   [Score: {p.score:8.4f}] -> {p.payload['text']} ({p.payload['category']})")

# === 6. 執行主程式 ===
if __name__ == "__main__":
    # 步驟 1: 動態初始化 (自動抓 Size)
    initialize_all_vdbs()

    # 步驟 2: 準備大批次測試資料
    test_data = [
        {"text": "Python 廣泛應用於人工智慧開發", "category": "tech"},
        {"text": "GPU 算力對於訓練大模型非常重要", "category": "tech"},
        {"text": "今日台北氣溫偏高，午後有雨", "category": "weather"},
        {"text": "這碗牛肉麵的湯頭濃郁，麵條Ｑ彈", "category": "food"}
    ]

    # 步驟 3: 執行同步批次上傳 (不再單點上傳)
    batch_upsert_to_all(test_data)

    # 步驟 4: 測試分類搜尋
    compare_search("AI 與程式語言", target_category="tech")
    compare_search("天氣預報", target_category="weather")