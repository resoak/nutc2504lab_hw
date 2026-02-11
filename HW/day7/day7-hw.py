import os
import uuid
import requests
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import re

# --- 環境設定 ---
BASE_CACHE = "C:/huggingface_cache"
os.makedirs(BASE_CACHE, exist_ok=True)
os.environ['HF_HOME'] = BASE_CACHE
os.environ['DOCLING_CACHE_DIR'] = BASE_CACHE
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# --- Docling 相關導入 ---
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions, 
    PdfPipelineOptions, 
    RapidOcrOptions
)
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

# --- LLM Guard / RAG / 評測導入 ---
from llm_guard.input_scanners import PromptInjection
from llm_guard.input_scanners.prompt_injection import MatchType
from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepeval.metrics import (
    FaithfulnessMetric, 
    AnswerRelevancyMetric, 
    ContextualRecallMetric, 
    ContextualPrecisionMetric
)
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI

# --- 1. 配置區域 ---
VLLM_URL = "https://ws-01.wade0426.me/v1/chat/completions"
EMBED_URL = "https://ws-04.wade0426.me/embed"
CHAT_API_URL = "https://ws-03.wade0426.me/v1"
JUDGE_MODEL_ID = "/models/Qwen3-30B-A3B-Instruct-2507-FP8"
COLLECTION_NAME = "final_hybrid_rag_stable"
RERANKER_PATH = r"C:\Users\RS\Downloads\Qwen3-Reranker-0.6B" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2

# Token 管理配置
MAX_CONTEXT_CHARS = 6000  # 字元限制（約 2000-2500 tokens）
MAX_CHUNK_SIZE = 500  # 每個語意塊的最大字元數
MAX_IMAGE_PIXELS = 800 * 800

# --- 2. 語意分塊工具 ---
class SemanticChunker:
    """
    基於語意的文本分塊器
    """
    def __init__(self, max_chunk_size=MAX_CHUNK_SIZE):
        self.max_chunk_size = max_chunk_size
    
    def split_by_semantics(self, text):
        """
        按照語意邊界分割文本
        優先級: 段落 > 句子 > 固定長度
        """
        if not text or len(text) < self.max_chunk_size:
            return [text] if text else []
        
        chunks = []
        
        # 第一步: 按段落分割（空行或換行符）
        paragraphs = re.split(r'\n\s*\n|\n{2,}', text)
        
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 如果當前段落本身太長，需要進一步分割
            if len(para) > self.max_chunk_size:
                # 先保存當前累積的塊
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # 按句子分割長段落
                sentences = self._split_sentences(para)
                for sent in sentences:
                    if len(current_chunk) + len(sent) > self.max_chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sent
                    else:
                        current_chunk += sent + " "
            else:
                # 正常段落，嘗試合併
                if len(current_chunk) + len(para) > self.max_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = para
                else:
                    current_chunk += "\n" + para if current_chunk else para
        
        # 保存最後的塊
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_sentences(self, text):
        """
        按句子分割文本
        """
        # 中英文句子邊界
        pattern = r'([。！？\.!?]+[\s"\'）】]*)'
        sentences = re.split(pattern, text)
        
        # 重組句子（包含標點）
        result = []
        for i in range(0, len(sentences)-1, 2):
            sent = sentences[i]
            if i+1 < len(sentences):
                sent += sentences[i+1]
            if sent.strip():
                result.append(sent.strip())
        
        # 處理最後一個可能沒有標點的句子
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())
        
        return result
    
    def chunk_document(self, document_text):
        """
        將整個文檔分塊
        """
        chunks = self.split_by_semantics(document_text)
        
        print(f"  📦 文檔分塊: {len(document_text)} 字元 → {len(chunks)} 個語意塊")
        
        return chunks

# --- 3. 圖片預處理工具 ---
def resize_image_if_needed(img_path, max_pixels=MAX_IMAGE_PIXELS):
    """
    如果圖片太大，調整大小以避免 VLM token 溢出
    """
    try:
        img = Image.open(img_path)
        width, height = img.size
        total_pixels = width * height
        
        print(f"  📐 圖片尺寸: {width}x{height} ({total_pixels:,} 像素)")
        
        if total_pixels > max_pixels:
            ratio = (max_pixels / total_pixels) ** 0.5
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            print(f"  🔄 調整大小至: {new_width}x{new_height}")
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            temp_path = f"temp_resized_{os.path.basename(img_path)}"
            img.save(temp_path, format=img.format or 'PNG')
            return temp_path
        
        return img_path
    except Exception as e:
        print(f"  ⚠️  圖片預處理失敗: {e}")
        return img_path

# --- 4. 改進的安全掃描器 ---
class FlexiblePDFScanner:
    """
    更靈活的安全掃描器
    """
    def __init__(self):
        self.scanner = PromptInjection(threshold=0.95, match_type=MatchType.SENTENCE)
        self.trusted_extensions = ['.docx', '.xlsx', '.pptx']
    
    def scan_content(self, content, file_name):
        print(f"[*] 正在掃描安全風險: {file_name}")
        
        if not content or len(content) < 100:
            print(f"  ℹ️  內容過短，直接通過")
            return True, 0.0
        
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # 信任的文檔類型使用更寬鬆的檢查
        if file_ext in self.trusted_extensions:
            print(f"  ℹ️  信任的文檔類型 ({file_ext})，使用寬鬆檢查")
            sections = [content[i:i+2000] for i in range(0, min(len(content), 6000), 2000)]
            unsafe_count = 0
            max_risk = 0.0
            
            for idx, s in enumerate(sections[:3]):
                _, is_safe, risk_score = self.scanner.scan(s)
                max_risk = max(max_risk, risk_score)
                if not is_safe:
                    unsafe_count += 1
                    print(f"    ⚠️  段落 {idx+1} 風險分數: {risk_score:.2f}")
            
            if unsafe_count >= 3:
                print(f"  🚨 檢測到 {unsafe_count} 個高風險段落")
                return False, max_risk
            
            print(f"  ✅ 安全檢查通過 (最高風險: {max_risk:.2f})")
            return True, max_risk
        
        # 一般文檔檢查
        sections = [content[i:i+1500] for i in range(0, len(content), 1500)]
        max_risk = 0.0
        unsafe_count = 0
        
        for idx, s in enumerate(sections[:5]):
            _, is_safe, risk_score = self.scanner.scan(s)
            max_risk = max(max_risk, risk_score)
            if not is_safe:
                unsafe_count += 1
                print(f"    ⚠️  段落 {idx+1} 風險分數: {risk_score:.2f}")
        
        if unsafe_count >= 2:
            print(f"  🚨 檢測到 {unsafe_count} 個高風險段落")
            return False, max_risk
        
        print(f"  ✅ 安全檢查通過 (最高風險: {max_risk:.2f})")
        return True, max_risk

# --- 5. 改進的解析器工廠 ---
def get_converters():
    """配置兩種轉換器"""
    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_ocr = True
    pdf_opts.do_table_structure = False
    pdf_opts.ocr_options = RapidOcrOptions() 

    standard_conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
    )
    
    vlm_opts = ApiVlmOptions(
        url=VLLM_URL,
        params=dict(
            model="allenai/olmOCR-2-7B-1025-FP8", 
            max_tokens=1500,
            temperature=0.1
        ),
        prompt="Extract all text from this image. Be concise.",
        response_format=ResponseFormat.MARKDOWN,
    )
    vlm_pipe_opts = VlmPipelineOptions(
        enable_remote_services=True, 
        vlm_options=vlm_opts
    )
    
    vlm_conv = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=vlm_pipe_opts, 
                pipeline_cls=VlmPipeline
            ),
            InputFormat.IMAGE: ImageFormatOption(
                pipeline_options=vlm_pipe_opts, 
                pipeline_cls=VlmPipeline
            )
        }
    )
    
    return standard_conv, vlm_conv

# --- 6. 智能上下文選擇器 ---
def select_best_chunks(candidate_objects, max_chars=MAX_CONTEXT_CHARS):
    """
    candidate_objects: list of dicts {text, score, source, orig_idx}
    """
    # 1. 先按分數從高到低排序，選取最相關的
    sorted_items = sorted(candidate_objects, key=lambda x: x["score"], reverse=True)
    
    selected = []
    total_chars = 0
    for item in sorted_items:
        if total_chars + len(item["text"]) <= max_chars:
            selected.append(item)
            total_chars += len(item["text"])
        else:
            # 如果這一個太長，可以跳過找下一個短一點的，或者直接 break
            continue
            
    # 2. 恢復原始順序 (重要：維持文檔閱讀的邏輯順序)
    selected.sort(key=lambda x: x["orig_idx"])
    
    return [s["text"] for s in selected], [s["source"] for s in selected], total_chars

# --- 7. DeepEval Judge (簡化版) ---
class SimpleJudge(DeepEvalBaseLLM):    
    def __init__(self, base_url=CHAT_API_URL, model_name=JUDGE_MODEL_ID):
        self.base_url = base_url
        self.model_name = model_name
    
    def load_model(self): 
        return OpenAI(api_key="NoNeed", base_url=self.base_url)
    
    def generate(self, prompt: str) -> str:
        client = self.load_model()
        
        # 嚴格限制 prompt 長度
        if len(prompt) > 7000:  # 字元限制
            print(f"  ⚠️  Prompt 過長 ({len(prompt)} 字元)，截斷到 7000")
            prompt = prompt[:7000] + "\n\n[內容已截斷]"
        
        try:
            response = client.chat.completions.create(
                model=self.model_name, 
                messages=[{"role": "user", "content": prompt}], 
                temperature=0.0,  # 降低溫度確保 JSON 輸出
                max_tokens=512
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM 調用失敗: {str(e)[:100]}")
            print("prompt:",prompt)
            return "生成失敗"
    
    async def a_generate(self, p): 
        return self.generate(p)
    
    def get_model_name(self): 
        return "Qwen3-Judge"

# --- 8. 文檔處理函數 ---
def process_document_with_fallback(filepath, standard_conv, vlm_conv, guard):
    """帶有完整錯誤處理的文檔處理"""
    filename = os.path.basename(filepath)
    file_ext = os.path.splitext(filename)[1].lower()
    
    try:
        if file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
            processed_path = resize_image_if_needed(filepath)
            filepath = processed_path
        
        if filename in ["1.pdf", "2.pdf"]:
            print(f"[*] 🚀 使用 RapidOCR 解析: {filename}")
            converter = standard_conv
            use_vlm = False
        else:
            print(f"[*] 🧠 使用 olmOCR 解析: {filename}")
            converter = vlm_conv
            use_vlm = True
        
        result = converter.convert(filepath)
        content = result.document.export_to_markdown()
        
        if not content or len(content.strip()) < 10:
            print(f"  ⚠️  {filename} 內容為空，嘗試備用方案")
            if use_vlm:
                print(f"  🔄 切換到 RapidOCR")
                result = standard_conv.convert(filepath)
                content = result.document.export_to_markdown()
            else:
                print(f"  🔄 切換到 olmOCR")
                result = vlm_conv.convert(filepath)
                content = result.document.export_to_markdown()
        
        is_safe, risk = guard.scan_content(content, filename)
        
        if is_safe:
            return content, filename, True
        else:
            print(f"  🚨 {filename} 風險分數過高")
            return None, filename, False
            
    except Exception as e:
        print(f"  ❌ {filename} 解析失敗: {e}")
        return None, filename, False
    finally:
        if file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
            temp_path = f"temp_resized_{filename}"
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

# --- 9. 主程式 ---
def main():
    print(">>> 載入 Reranker 模型...")
    reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_PATH, local_files_only=True)
    reranker_model = AutoModelForCausalLM.from_pretrained(
        RERANKER_PATH, local_files_only=True, torch_dtype=torch.float16
    ).to(DEVICE).eval()
    
    token_no = reranker_tokenizer.convert_tokens_to_ids("no")
    token_yes = reranker_tokenizer.convert_tokens_to_ids("yes")
    
    q_client = QdrantClient(host="localhost", port=6333)
    guard = FlexiblePDFScanner()
    custom_llm = SimpleJudge()
    chunker = SemanticChunker(max_chunk_size=MAX_CHUNK_SIZE)
    
    standard_conv, vlm_conv = get_converters()

    # --- 第一階段: 文件解析與分塊 ---
    print("\n\033[94m>>> [第一階段: 文件解析與語意分塊]\033[0m")
    target_files = ["1.pdf","2.pdf","3.pdf","4.png","5.docx"]
    all_chunks = []
    all_metas = []
    
    success_count = 0
    fail_count = 0

    for f in target_files:
        if not os.path.exists(f): 
            print(f"  ⚠️  檔案不存在: {f}")
            continue
        
        content, filename, success = process_document_with_fallback(
            f, standard_conv, vlm_conv, guard
        )
        
        if success and content:
            # 使用語意分塊
            chunks = chunker.chunk_document(content)
            
            # 為每個塊添加來源信息
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metas.append(filename)
            
            success_count += 1
            print(f"  ✅ {filename} 處理成功 → {len(chunks)} 個塊")
        else:
            fail_count += 1

    print(f"\n📊 處理統計: 成功 {success_count} 個文檔, 失敗 {fail_count} 個")
    print(f"📦 總塊數: {len(all_chunks)} 個語意塊")
    
    # --- 第二階段: Qdrant 入庫 ---
    if not all_chunks: 
        print("❌ 沒有任何內容，程式結束")
        return
    
    def get_embs(texts):
        response = requests.post(
            EMBED_URL, 
            json={"texts": texts, "task_description": "檢索", "normalize": True}
        )
        return response.json()["embeddings"]
    
    dim = len(get_embs(["test"])[0])
    print(f"\n[*] 向量維度: {dim}")
    
    if q_client.collection_exists(COLLECTION_NAME): 
        q_client.delete_collection(COLLECTION_NAME)
    
    q_client.create_collection(
        COLLECTION_NAME, 
        vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)}, 
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)}
    )
    
    print(f"[*] 正在產生向量並入庫...")
    
    # 批次生成向量
    all_embs = []
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        all_embs.extend(get_embs(batch))
    
    # 建立索引點
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()), 
            vector={
                "dense": emb, 
                "sparse": models.Document(text=chunk, model="Qdrant/bm25")
            }, 
            payload={"text": chunk, "source": source}
        ) 
        for chunk, emb, source in zip(all_chunks, all_embs, all_metas)
    ]
    
    q_client.upsert(COLLECTION_NAME, points, wait=True)
    print(f"✅ 入庫完成")

    # --- 第三階段: RAG 執行與評測 ---
    print("\n\033[94m>>> [第二階段: RAG 執行]\033[0m")
    
    if not os.path.exists('questions.csv'):
        print("❌ questions.csv 不存在")
        return
    
    q_df = pd.read_csv('questions.csv').head(5)
    ans_df = pd.read_csv('questions_answer.csv')
    
    # 簡化評測指標（減少 API 調用）
    metrics = {
        "Relevancy": AnswerRelevancyMetric(model=custom_llm,include_reason=True),
        "Faith": FaithfulnessMetric(model=custom_llm,include_reason=True),
        "Precision":ContextualPrecisionMetric(model=custom_llm,include_reason=True),
        "Recall":ContextualRecallMetric(model=custom_llm,include_reason=True)
    }
    
    final_output = []
    
    for idx, row in q_df.iterrows():
        try:
            qid = str(row['id'])
            qtxt = str(row['questions'])
            
            g_truth_rows = ans_df[ans_df['id'].astype(str) == qid]
            if len(g_truth_rows) == 0:
                print(f"⚠️  ID {qid} 沒有對應答案")
                continue
            g_truth = str(g_truth_rows['answer'].values[0])

            q_emb = get_embs([qtxt])[0]
            
            # 混合檢索 - 增加檢索數量
            search_res = q_client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    models.Prefetch(
                        query=models.Document(text=qtxt, model="Qdrant/bm25"), 
                        using="sparse", 
                        limit=20
                    ),
                    models.Prefetch(query=q_emb, using="dense", limit=20),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=15  # 獲取更多候選
            )

            candidates = [p.payload["text"] for p in search_res.points]
            if not candidates: 
                print(f"⚠️  ID {qid} 沒有檢索結果")
                continue
            # ... 前面檢索部分不變 ...
            candidates_text = [p.payload["text"] for p in search_res.points]
            candidates_source = [p.payload["source"] for p in search_res.points]
            
            if not candidates_text: 
                print(f"⚠️  ID {qid} 沒有檢索結果")
                continue
            
            # Rerank
            rerank_pairs = [[qtxt, c] for c in candidates_text]
            inputs = reranker_tokenizer(
                rerank_pairs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt',
                max_length=512
            ).to(DEVICE)
            
            with torch.no_grad():
                logits = reranker_model(**inputs).logits[:, -1, [token_no, token_yes]]
                scores = torch.softmax(logits, dim=-1)[:, 1].tolist()
            
            # --- 關鍵修正：封裝物件以便 select_best_chunks 處理 ---
            candidate_objs = []
            for i in range(len(candidates_text)):
                candidate_objs.append({
                    "text": candidates_text[i],
                    "score": scores[i],
                    "source": candidates_source[i],
                    "orig_idx": i  # 這裡保留檢索出來的原始順序
                })
            
            # 智能選擇最佳塊
            best_chunks, best_sources_list, total_chars = select_best_chunks(candidate_objs, MAX_CONTEXT_CHARS)
            best_sources = list(set(best_sources_list))
            
            print(f"\n>> ID {qid}:")
            print(f"  📊 選中 {len(best_chunks)} 個塊，共 {total_chars} 字元")
            
            # 生成答案
            context_text = "\n\n".join(best_chunks)
            prompt = f"請根據以下資料回答問題。\n\n資料：\n{context_text}\n\n問題：{qtxt}\n\n答案："
            
            # ... 後續生成與評測邏輯 ...
        
            
            print(f"  📝 Prompt 長度: {len(prompt)} 字元")
            
            ans = custom_llm.generate(prompt)
            print("ans:", ans)
            # 簡化評測（只評關鍵指標）
            print(f"  評測結果:")
            tc = LLMTestCase(
                input=qtxt, 
                actual_output=ans, 
                retrieval_context=best_chunks,
                expected_output=g_truth
            )
            
            # 建立這一列的結果紀錄
            row_result = {
                'q_id': qid, 
                'questions': qtxt, 
                'answer': ans, 
                'source': ",".join(best_sources)
            }

            print(f"  評測結果:")
            for name, m in metrics.items():
                try:
                    m.measure(tc)
                    print(f"   [*] {name}: {m.score:.2f}")
                    # 將分數與原因存入該列字典
                    row_result[f'{name}_score'] = m.score
                    row_result[f'{name}_reason'] = getattr(m, 'reason', 'N/A')
                except Exception as e:
                    print(f"   [!] {name} 評測失敗: {e}")
                    row_result[f'{name}_score'] = 0
                    row_result[f'{name}_reason'] = "Error"

            final_output.append(row_result)
            
        except Exception as e: 
            print(f"❌ ID {qid} 失敗: {e}")

    # 儲存結果
    # if final_output:
    #     pd.DataFrame(final_output).to_csv(
    #         'test_dataset_final.csv', 
    #         index=False, 
    #         encoding='utf-8-sig'
    #     )
    #     print(f"\n✅ 完成！共生成 {len(final_output)} 個答案")
    # else:
    #     print("\n⚠️  沒有生成答案")
    # --- 第四階段: 獨立處理 test_dataset.csv (無評測模式) ---
    if os.path.exists('test_dataset.csv'):
        print("\n\033[92m>>> [第四階段: 處理 test_dataset.csv]\033[0m")
        test_df = pd.read_csv('test_dataset.csv')
        test_final_output = []

        for _, row in test_df.iterrows():
            try:
                tid = str(row.get('id', 'N/A'))
                tqtxt = str(row.get('questions', ''))
                if not tqtxt: continue

                print(f"[*] 正在處理測試集 ID: {tid}...", end="\r")

                # 1. 檢索
                t_q_emb = get_embs([tqtxt])[0]
                t_search_res = q_client.query_points(
                    collection_name=COLLECTION_NAME,
                    prefetch=[
                        models.Prefetch(query=models.Document(text=tqtxt, model="Qdrant/bm25"), using="sparse", limit=20),
                        models.Prefetch(query=t_q_emb, using="dense", limit=20),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=15
                )

                t_cand_text = [p.payload["text"] for p in t_search_res.points]
                t_cand_src = [p.payload["source"] for p in t_search_res.points]

                if not t_cand_text:
                    test_final_output.append({'id': tid, 'questions': tqtxt, 'answer': "查無資料", 'source': ""})
                    continue

                # 2. Rerank
                t_rerank_pairs = [[tqtxt, c] for c in t_cand_text]
                t_inputs = reranker_tokenizer(t_rerank_pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(DEVICE)
                with torch.no_grad():
                    t_logits = reranker_model(**t_inputs).logits[:, -1, [token_no, token_yes]]
                    t_scores = torch.softmax(t_logits, dim=-1)[:, 1].tolist()

                # 3. 智能選擇
                t_objs = []
                for i in range(len(t_cand_text)):
                    t_objs.append({
                        "text": t_cand_text[i],
                        "score": t_scores[i],
                        "source": t_cand_src[i],
                        "orig_idx": i
                    })
                
                t_best_chunks, t_best_srcs, _ = select_best_chunks(t_objs, MAX_CONTEXT_CHARS)

                # 4. 生成答案
                t_context = "\n\n".join(t_best_chunks)
                t_prompt = f"請根據以下資料回答問題。\n\n資料：\n{t_context}\n\n問題：{tqtxt}\n\n答案："
                t_ans = custom_llm.generate(t_prompt)

                test_final_output.append({
                    'id': tid,
                    'questions': tqtxt,
                    'answer': t_ans,
                    'source': ",".join(list(set(t_best_srcs)))
                })

            except Exception as e:
                print(f"\n❌ 測試集 ID {tid} 失敗: {e}")

        # 儲存測試集結果
        if test_final_output:
            pd.DataFrame(test_final_output).to_csv('test_dataset_final.csv', index=False, encoding='utf-8-sig')
            print(f"\n✅ 測試集處理完成，已存至 test_dataset_final.csv")
    else:
        print("\nℹ️ 未發現 test_dataset.csv，跳過此階段。")

if __name__ == "__main__":
    main()