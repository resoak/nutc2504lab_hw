import os
import uuid
import requests
import pandas as pd
import torch
import numpy as np
from pathlib import Path

# Docling 核心
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions, PdfPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

# LLM Guard 相關
from llm_guard.input_scanners import PromptInjection
from llm_guard.input_scanners.prompt_injection import MatchType

# RAG 與評測
from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepeval.metrics import (
    FaithfulnessMetric, AnswerRelevancyMetric, 
    ContextualRecallMetric, ContextualPrecisionMetric, ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI

# --- 1. 配置區域 ---
VLLM_URL = "https://ws-01.wade0426.me/v1/chat/completions"
EMBED_URL = "https://ws-04.wade0426.me/embed"
CHAT_API_URL = "https://ws-03.wade0426.me/v1"
# 使用 API 指向的 Judge Model ID
JUDGE_MODEL_ID = "/models/Qwen3-30B-A3B-Instruct-2507-FP8"

COLLECTION_NAME = "final_secure_rag_v15"
RERANKER_PATH = os.path.expanduser("~/AI/Models/Qwen3-Reranker-0.6B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K = 10
BATCH_SIZE = 2

# --- 2. 安全掃描器類別 (整合你的 SimplePDFScanner 邏輯) ---

class SimplePDFScanner:
    def __init__(self):
        # 門檻值設為 0.3，匹配類型為句子
        self.scanner = PromptInjection(threshold=0.95, match_type=MatchType.SENTENCE)
   
    def scan_content(self, content, file_name):
        """掃描內容並返回是否安全"""
        print(f"\n{'-'*30}")
        print(f"[*] 正在掃描安全風險: {file_name}")
       
        sections = self._split_content(content)
        detections = []
        max_risk = 0.0
        total_sections = len(sections)
       
        for i, section in enumerate(sections, 1):
            # 執行 LLM Guard 掃描
            _, is_safe, risk_score = self.scanner.scan(section)
           
            if not is_safe or risk_score > 0.9:
                detections.append({'section': i, 'risk_score': risk_score})
                max_risk = max(max_risk, risk_score)
                print(f"  [!] 第 {i}/{total_sections} 段: 風險 {risk_score:.2f} (未通過)")
       
        is_safe_final = len(detections) == 0
        status = "\033[92m[PASS]\033[0m" if is_safe_final else "\033[91m[FAIL]\033[0m"
        print(f"[*] 掃描結果: {status} | 最高風險分數: {max_risk:.2f}")
        return is_safe_final, max_risk

    def _split_content(self, content, chunk_size=1000):
        """將內容分段以提高檢測率"""
        paragraphs = content.split('\n\n')
        sections, current = [], ""
        for para in paragraphs:
            if len(current) + len(para) > chunk_size:
                if current: sections.append(current.strip())
                current = para
            else:
                current += "\n\n" + para if current else para
        if current: sections.append(current.strip())
        if not sections: sections = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        return [s for s in sections if len(s.strip()) > 50]

def olmocr2_vlm_options(
    model: str = "allenai/olmOCR-2-7B-1025-FP8",
    hostname_and_port: str = "https://ws-01.wade0426.me/v1/",
    prompt: str = "Convert this page to markdown.",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    api_key: str = "",) -> ApiVlmOptions:


    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
   
    options = ApiVlmOptions(
        url=f"{hostname_and_port}",
        params=dict(
            model=model,
            max_tokens=max_tokens,
        ),
        headers=headers,
        prompt=prompt,
        timeout=120,  # olmocr2 可能需要較長處理時間
        scale=0.8,  # 圖片縮放比例
        temperature=temperature,
        response_format=ResponseFormat.MARKDOWN,
    )
    return options


# --- 3. 解析器工廠 ---
def get_converters():
    standard_conv = DocumentConverter() 
    
    # vlm_opts = ApiVlmOptions(
    #     url=f"{VLLM_URL}",
    #     params=dict(model="allenai/olmOCR-2-7B-1025-FP8", max_tokens=4096),
    #     prompt="Convert this page to clean, readable markdown format.",
    #     timeout=240, scale=1.0, temperature=0.0,
    #     response_format=ResponseFormat.MARKDOWN,
    # )
    
    vlm_pipe_opts = VlmPipelineOptions(enable_remote_services=True)
    vlm_pipe_opts.vlm_options = olmocr2_vlm_options(
            model="allenai/olmOCR-2-7B-1025-FP8",
            hostname_and_port=VLLM_URL,
            prompt="Convert this page to clean, readable markdown format.",
            temperature=0.0,  # olmocr2 建議使用較低的溫度
        )

    
    vlm_conv = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=vlm_pipe_opts, pipeline_cls=VlmPipeline),
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=vlm_pipe_opts, pipeline_cls=VlmPipeline)
        }
    )

    return standard_conv, vlm_conv

# --- 4. DeepEval 自定義評測模型 ---

class CustomJudge(DeepEvalBaseLLM):    
    def __init__(
        self,
        base_url=CHAT_API_URL,
        model_name=JUDGE_MODEL_ID
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

        structured_prompt = (
            f"{prompt}\n\n"
            "Respond ONLY in valid JSON format with the following keys:\n"
            "- \"verdict\": (string) either 'yes' or 'no'\n"
            "- \"reason\": (string) explaining your decision\n"
            "Ensure the keys are in English and the format is strictly valid JSON."
        )

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                # {"role": "system", "content": structured_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content
   
    async def a_generate(self, prompt: str) -> str:
        # 如果需要非同步版本，可以使用 AsyncOpenAI
        # 這裡為簡化示範，直接重用同步方法
        return self.generate(prompt)
   
    def get_model_name(self):
        return f"Llama.cpp ({self.model_name})"

# --- 5. 主程式 ---

def main():
    # 初始化 Reranker 模型
    global reranker_tokenizer, reranker_model, token_no, token_yes
    print(">>> 載入本地 Reranker 模型...")
    reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_PATH, local_files_only=True)
    reranker_model = AutoModelForCausalLM.from_pretrained(RERANKER_PATH, local_files_only=True, torch_dtype=torch.float16).to(DEVICE).eval()
    token_no, token_yes = reranker_tokenizer.convert_tokens_to_ids("no"), reranker_tokenizer.convert_tokens_to_ids("yes")
    
    q_client = QdrantClient(host="localhost", port=6333)
    guard = SimplePDFScanner()

    custom_llm = CustomJudge(
        base_url="https://ws-03.wade0426.me/v1",
        model_name="/models/Qwen3-30B-A3B-Instruct-2507-FP8"
    )

    # judge = CustomJudge()

    standard_conv, vlm_conv = get_converters()

    # --- 第一階段: 解析與安全掃描 ---
    print("\n\033[94m>>> [第一階段: 文件解析與安全檢測]\033[0m")
    files = ["4.png"]
    # files = ["1.pdf", "2.pdf", "3.pdf", "4.png", "5.docx"]
    docs_to_store, metas_to_store = [], []

    for f in files:
        if not os.path.exists(f): continue
        try:
            print(f"[*] 處理檔案: {f}", end="\r")
            if f in ["1.pdf", "2.pdf"]:
                res = standard_conv.convert(f)
            else:
                try: res = vlm_conv.convert(f)
                except Exception as e:
                    print("VLM 失敗時降級", e)
                    res = standard_conv.convert(f) # VLM 失敗時降級
            
            content = res.document.export_to_markdown()
            print("content",content)
            # 使用你的分段掃描邏輯進行安全檢查
            is_safe, risk = guard.scan_content(content, f)
            
            docs_to_store.append(content)
            metas_to_store.append(f)
            
            # if is_safe:
            #     docs_to_store.append(content)
            #     metas_to_store.append(f)
            # else:
            #     print(f"    - ❌ {f}: 偵測到 Prompt Injection 風險，已排除。")
        except Exception as e: print(f"    - ❌ {f} 解析失敗: {e}")

    # --- 第二階段: Qdrant 入庫 ---
    if docs_to_store:
        def get_embs(t): return requests.post(EMBED_URL, json={"texts": t, "task_description": "檢索", "normalize": True}).json()["embeddings"]
        dim = len(get_embs(["test"])[0])
        if q_client.collection_exists(COLLECTION_NAME): q_client.delete_collection(COLLECTION_NAME)
        q_client.create_collection(COLLECTION_NAME, vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)}, sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)})
        
        all_embs = []
        for i in range(0, len(docs_to_store), BATCH_SIZE):
            all_embs.extend(get_embs(docs_to_store[i : i + BATCH_SIZE]))
        
        points = [models.PointStruct(id=str(uuid.uuid4()), vector={"dense": e, "sparse": models.Document(text=t, model="Qdrant/bm25")}, payload={"text": t, "source": s}) for t, e, s in zip(docs_to_store, all_embs, metas_to_store)]
        q_client.upsert(COLLECTION_NAME, points, wait=True)

    # --- 第三階段: 評測與 CSV 產出 ---
    print("\n\033[94m>>> [第二階段: RAG 執行與評測]\033[0m")
    q_df = pd.read_csv('questions.csv').rename(columns=lambda x: x.strip())
    ans_df = pd.read_csv('questions_answer.csv').rename(columns=lambda x: x.strip())
    
    metrics = {
        "F": FaithfulnessMetric(model=custom_llm), 
        "AR": AnswerRelevancyMetric(model=custom_llm), 
        "CR": ContextualRecallMetric(model=custom_llm), 
        "CP": ContextualPrecisionMetric(model=custom_llm), 
        "Crel": ContextualRelevancyMetric(model=custom_llm)
    }
    results = []

    def get_embs(t): return requests.post(EMBED_URL, json={"texts": t, "task_description": "檢索", "normalize": True}).json()["embeddings"]

    for _, row in q_df.iterrows():
        try:
            qid, qtxt = str(row['id']), str(row['questions'])
            match = ans_df[ans_df['id'].astype(str) == qid]
            g_truth = str(match['answer'].values[0]) if not match.empty else ""

            # 1. Hybrid 檢索 (Dense + Sparse)
            q_emb = get_embs([qtxt])[0]
            search_res = q_client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    models.Prefetch(query=models.Document(text=qtxt, model="Qdrant/bm25"), using="sparse", limit=10),
                    models.Prefetch(query=q_emb, using="dense", limit=10),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=10
            )
            print("search_res", search_res)

            # 2. Rerank 邏輯
            candidates = [p.payload["text"] for p in search_res.points]
            if not candidates:
                print(f"[-] ID {qid} 找不到相關文檔")
                continue

            rerank_pairs = [[qtxt, c] for c in candidates]
            inputs = reranker_tokenizer(rerank_pairs, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                logits = reranker_model(**inputs).logits[:, -1, [token_no, token_yes]]
                scores = torch.softmax(logits, dim=-1)[:, 1].tolist()
            
            ranked = sorted(zip(search_res.points, scores), key=lambda x: x[1], reverse=True)
            top_contexts = [r[0].payload["text"] for r in ranked[:3]]
            top_sources = ",".join(list(set([r[0].payload["source"] for r in ranked[:3]])))

            # 3. 生成回答
            ans = custom_llm.generate(f"Context: {' '.join(top_contexts)}\nQuestion: {qtxt}")
            
            print("ranked", ranked) # TO-DO
            print("input",qtxt)
            print("actual_output",ans)
            print("retrieval_context",top_contexts)
            print("expected_output",g_truth)

            # 4. 執行 DeepEval 指標測量
            test_case = LLMTestCase(
                input=qtxt, 
                actual_output=ans, 
                retrieval_context=top_contexts,
                expected_output=g_truth
            )
            
            res_row = {'id': qid, 'questions': qtxt, 'answer': ans, 'source': top_sources}
            
            # 將每個 metric 的分數塞進 res_row
            for key, metric in metrics.items():
                metric.measure(test_case)
                res_row[key] = metric.score
            
            results.append(res_row)
            print(f"✅ ID {qid} 評測完成 (F: {res_row['F']:.2f}, AR: {res_row['AR']:.2f})")
            
        except Exception as e: 
            print(f"❌ ID {qid} 發生錯誤: {e}")

    # 儲存結果
    pd.DataFrame(results).to_csv('day6_HW_questions.csv', index=False, encoding='utf-8-sig')
    print("\n>>> 任務完成！結果已儲存至 day6_HW_questions.csv")

if __name__ == "__main__":
    main()