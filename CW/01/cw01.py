import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 1. 建立 Qdrant 連接與 Collection
client = QdrantClient(url="http://localhost:6333")
collection_name = "my_homework_collection"

# 建立 Collection (如果已經存在會報錯，實務上可先 check 或使用 try-except)
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
)

# 定義一個輔助函式：使用 API 獲得向量
def get_embeddings(texts):
    url = "https://ws-04.wade0426.me/embed"
    data = {
        "texts": texts,
        "normalize": True,
        "batch_size": 32
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()['embeddings']
    else:
        raise Exception(f"API Error: {response.text}")

# 2. 準備五個（或更多）Point 的資料
raw_texts = [
    "人工智慧很有趣",
    "機器學習是 AI 的一個分支",
    "向量資料庫適合儲存非結構化資料",
    "Python 是開發 AI 的熱門語言",
    "台北今天的氣候很晴朗",
    "大模型改變了搜尋的方式"
]

# 3. 使用 API 獲得向量
print("正在取得向量...")
vectors = get_embeddings(raw_texts)

# 4. 嵌入到 VDB (Upsert points)
print("正在將資料存入 Qdrant...")
points = []
for i, (text, vec) in enumerate(zip(raw_texts, vectors)):
    points.append(
        PointStruct(
            id=i + 1,
            vector=vec,
            payload={"text": text, "source": "homework_01"}
        )
    )

client.upsert(
    collection_name=collection_name,
    points=points
)

# 5. 召回內容 (搜尋)
query_text = ["人工智慧是什麼"]
query_vector = get_embeddings(query_text)[0]

print("\n--- 搜尋結果 (召回內容) ---")
search_result = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=3
)

for point in search_result.points:
    print(f"ID: {point.id}")
    print(f"相似度分數 (Score): {point.score:.4f}")
    print(f"內容: {point.payload['text']}")
    print("-" * 20)