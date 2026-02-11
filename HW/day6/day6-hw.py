import pandas as pd
import requests
import json
import re
import os
import uuid
import torch
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepeval.metrics import (
    FaithfulnessMetric, 
    AnswerRelevancyMetric, 
    ContextualRecallMetric, 
    ContextualPrecisionMetric, 
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM

# --- 1. 配置區域 ---
CHAT_API_URL = "https://ws-03.wade0426.me/v1/chat/completions"
EMBED_URL = "https://ws-04.wade0426.me/embed"
MODEL_NAME = "test"
# RERANKER_PATH = os.path.expanduser("~/AI/Models/Qwen3-Reranker-0.6B")
COLLECTION_NAME = "day6_hybrid_search_demo"
MAX_WORKERS = 1  # 評測時的併行執行緒數

# --- 2. 工具函數 ---

def get_embeddings(texts):
    """批次取得 Embedding"""
    payload = {"texts": texts, "task_description": "檢索技術文件", "normalize": True}
    try:
        res = requests.post(EMBED_URL, json=payload, timeout=120)
        return res.json().get("embeddings", [])
    except Exception as e:
        print(f"Embedding 錯誤: {e}")
        return []

def llm_generate(prompt):
    """呼叫 LLM 生成答案"""
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }
    try:
        res = requests.post(CHAT_API_URL, json=payload, timeout=300)
        return res.json()['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        return ""

# --- 3. 第一階段：處理 Embedding 與 Qdrant 入庫 ---

client = QdrantClient(host="localhost", port=6333)

def stage_1_ingestion():
    if not os.path.exists('qa_data.txt'):
        print("找不到 qa_data.txt，停止入庫。")
        return

    with open('qa_data.txt', 'r', encoding='utf-8') as f:
        docs = [l.strip() for l in f.readlines() if l.strip()]
    
    print(f"\n>>> [階段 1] 開始處理 Embedding，共 {len(docs)} 筆...")
    
    # 批次取得向量 (避免 Payload 過大)
    all_embeddings = []
    batch_size = 25
    for i in range(0, len(docs), batch_size):
        chunk = docs[i : i + batch_size]
        embs = get_embeddings(chunk)
        all_embeddings.extend(embs)
        print(f"進度: {len(all_embeddings)}/{len(docs)}")

    if not all_embeddings: return

    # 初始化 Collection
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": models.VectorParams(size=len(all_embeddings[0]), distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)}
    )
    
    # 批次 Upsert 到 Qdrant
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector={"dense": emb, "sparse": models.Document(text=doc, model="Qdrant/bm25")},
            payload={"text": doc}
        ) for doc, emb in zip(docs, all_embeddings)
    ]
    
    q_batch = 50
    for i in range(0, len(points), q_batch):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i : i + q_batch])
    
    print(">>> 階段 1 完成：向量資料庫已就緒。")

# --- 4. 載入本地 Reranker (優化批次推理) ---

model_path = os.path.expanduser("~/AI/Models/Qwen3-Reranker-0.6B")

# 設定設備
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model from: {model_path}...")
reranker_tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    local_files_only=True, 
    trust_remote_code=True
)

reranker_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device).eval()

# 獲取 "yes" 和 "no" 的 Token ID
token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")
token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")

# 設定最大長度與 Prompt 模板
max_reranker_length = 8192
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n"

prefix_tokens = reranker_tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = reranker_tokenizer.encode(suffix, add_special_tokens=False)

print("Reranker loaded successfully.")

# --- 2. 功能函式定義 ---

def format_instruction(instruction, query, doc):
    """格式化 reranker 的內容輸入"""
    if instruction is None:
        instruction = '根據查詢檢索相關文件'
    
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

def get_rerank_scores(query, documents, instruction=None):
    """
    輸入 query 和多份 documents，回傳每份文件的相關性分數
    """
    # 建立配對字串
    pairs = [format_instruction(instruction, query, doc) for doc in documents]
    
    processed_texts = []
    for pair in pairs:
        # 計算中間內容的 token，保留 prefix/suffix 的空間
        pair_ids = reranker_tokenizer.encode(
            pair,
            add_special_tokens=False,
            truncation=True,
            max_length=max_reranker_length - len(prefix_tokens) - len(suffix_tokens)
        )
        # 組合完整 Input
        full_ids = prefix_tokens + pair_ids + suffix_tokens
        processed_texts.append(reranker_tokenizer.decode(full_ids))

    # Tokenize 並推送到模型
    inputs = reranker_tokenizer(
        processed_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_reranker_length
    ).to(device)

    # 執行推理 (不計算梯度)
    with torch.no_grad():
        outputs = reranker_model(**inputs)
        # 取得最後一個位置的 Logits (預測下一個字是 yes 還是 no)
        # logits shape: [batch_size, sequence_length, vocab_size]
        last_token_logits = outputs.logits[:, -1, :]
        
        # 提取 yes 和 no 的分數
        target_logits = last_token_logits[:, [token_false_id, token_true_id]]
        # 使用 Softmax 轉換為機率，取第二欄 (yes) 作為相關度分數
        scores = torch.softmax(target_logits, dim=-1)[:, 1].cpu().detach().numpy()

    return scores

@torch.no_grad()
def rerank_best(query, documents):
    scores = get_rerank_scores(query, documents)
    results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return results


# --- 5. 第二階段：執行 RAG 並最後進行 DeepEval ---

class CustomJudge(DeepEvalBaseLLM):
    def __init__(
        self,
        base_url="https://ws-03.wade0426.me/v1/chat/completions",
        model_name="/models/Qwen3-30B-A3B-Instruct-2507-FP8"
    ):
        self.base_url = base_url
        self.model_name = model_name
       
    def load_model(self):
        # 建立 OpenAI 客戶端
        return OpenAI(
            api_key="NoNeed",
            base_url=self.base_url
        )
   
    def generate(self, prompt: str) -> str:
        client = self.load_model()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
   
    async def a_generate(self, prompt: str) -> str:
        # 如果需要非同步版本，可以使用 AsyncOpenAI
        # 這裡為簡化示範，直接重用同步方法
        return self.generate(prompt)
   
    def get_model_name(self):
        return f"Llama.cpp ({self.model_name})"


def run_rag_and_eval(q_id, q_text, ground_truth, metrics):
    """執行 RAG 流程並直接計算指標"""
    # 1. 檢索
    q_emb = get_embeddings([q_text])[0]
    res = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=models.Document(text=q_text, model="Qdrant/bm25"), using="sparse", limit=10),
            models.Prefetch(query=q_emb, using="dense", limit=10),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=10
    )
    candidates = [p.payload["text"] for p in res.points]
    
    # 2. Rerank
    best_doc = rerank_best(q_text, candidates)
    # print("best_doc:", best_doc)

    context_RAG = ""
    for i in best_doc:
        context_RAG += i[0]
    
    best_doc = [i[0] for i in best_doc]
    
    print(context_RAG,"\n\n")

    # 3. 生成回答
    prompt = f"請根據以下資料回答問題。\n資料：{context_RAG}\n問題：{q_text}"
    answer = llm_generate(prompt)
    print("LLM:", answer)
    
    # 4. DeepEval 評測
    test_case = LLMTestCase(input=q_text, actual_output=answer, retrieval_context=best_doc, expected_output=ground_truth)
    scores = {}
    for k, m in metrics.items():
        try: m.measure(test_case); scores[k] = m.score
        except Exception as e:print(e)
        
    return {
        'q_id': q_id, 'questions': q_text, 'answer': answer,
        'Faithfulness（忠實度）': scores['F'], 'Answer_Relevancy（答案相關性）': scores['AR'],
        'Contextual_Recall（上下文召回率）': scores['CR'], 'Contextual_Precision（上下文精確度）': scores['CP'],
        'Contextual_Relevancy（上下文相關性）': scores['Crel']
    }

def main():
    # --- 執行階段 1：處理 Embedding 並存入 Qdrant ---
    # stage_1_ingestion()

    # --- 執行階段 2：大批量處理 RAG + DeepEval ---
    q_df = pd.read_csv('questions.csv')
    truth_df = pd.read_csv('questions_answer.csv')
    judge = CustomJudge(
            base_url="https://ws-03.wade0426.me/v1",
            model_name="/models/Qwen3-30B-A3B-Instruct-2507-FP8"
        )
    
    metrics = {
        "F": FaithfulnessMetric(model=judge), "AR": AnswerRelevancyMetric(model=judge),
        "CR": ContextualRecallMetric(model=judge), "CP": ContextualPrecisionMetric(model=judge),
        "Crel": ContextualRelevancyMetric(model=judge)
    }

    final_results = []
    print(f"\n>>> [階段 2] 開始 RAG 檢索與 DeepEval 評測 (併行數: {MAX_WORKERS})...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for _, row in q_df.iterrows():
            g_truth = truth_df[truth_df['q_id'] == row['q_id']]['answer'].values[0]
            futures.append(executor.submit(run_rag_and_eval, row['q_id'], row['questions'], g_truth, metrics))
        
        for future in as_completed(futures):
            res = future.result()
            final_results.append(res)
            print(f"已完成 Q_ID: {res['q_id']}")

    # 儲存結果
    pd.DataFrame(final_results).sort_values('q_id').to_csv('day6_HW_questions.csv', index=False, encoding='utf-8-sig')
    print("\n>>> 所有流程已完成！結果已儲存至 day6_HW_questions.csv")

if __name__ == "__main__":
    main()