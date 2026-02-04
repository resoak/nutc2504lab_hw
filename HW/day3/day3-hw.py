import time
import requests
import json
from pathlib import Path
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END

# ==========================================
# 1. 基礎設定與 ASR 函數 (整合你提供的腳本)
# ==========================================
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="TOKEN", # 請替換為正確的 API Key
    model="google/gemma-3-27b-it",
    temperature=0
)

def run_asr_task(wav_path: str):
    """執行 ASR 轉錄並回傳 TXT 與 SRT 內容"""
    BASE = "https://3090api.huannago.com"
    CREATE_URL = f"{BASE}/api/v1/subtitle/tasks"
    auth = ("nutc2504", "nutc2504")

    # 1) 建立任務
    with open(wav_path, "rb") as f:
        r = requests.post(CREATE_URL, files={"audio": f}, timeout=60, auth=auth)
    r.raise_for_status()
    task_id = r.json()["id"]
    
    txt_url = f"{BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type=TXT" 
    srt_url = f"{BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type=SRT"

    def wait_download(url: str):
        for _ in range(300): # 最多等 10 分鐘
            resp = requests.get(url, auth=auth)
            if resp.status_code == 200: return resp.text
            time.sleep(2)
        return None

    print(f"--- ASR 任務 {task_id} 處理中 ---")
    return wait_download(txt_url), wait_download(srt_url)

# ==========================================
# 2. 定義 LangGraph State (狀態)
# ==========================================
class AssistantState(TypedDict):
    audio_path: str
    txt_content: str  # 純文字
    srt_content: str  # 時間軸文字
    minutes: str      # 整理後的會議記錄
    summary: str      # 重點摘要
    final_report: str # 最終報告

# ==========================================
# 3. 定義節點 (Nodes)
# ==========================================

def asr_node(state: AssistantState):
    """節點 1: ASR 語音轉文字"""
    txt, srt = run_asr_task(state["audio_path"])
    return {"txt_content": txt, "srt_content": srt}

def minutes_taker_node(state: AssistantState):
    """節點 2: 整理詳細逐字稿 (使用 SRT)"""
    prompt = f"請根據以下帶有時間軸的 SRT 內容，整理出一份格式整齊、易於閱讀的會議逐字稿：\n\n{state['srt_content']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"minutes": response.content}

def summarizer_node(state: AssistantState):
    """節點 3: 產生重點摘要 (使用 TXT)"""
    prompt = f"請針對以下會議內容，精煉出 3 到 5 個核心重點摘要：\n\n{state['txt_content']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"summary": response.content}

def writer_node(state: AssistantState):
    """節點 4: 彙整最終報告"""
    report = f"""
# 智慧會議記錄報告

## 一、 重點摘要
{state['summary']}

---

## 二、 詳細逐字稿內容
{state['minutes']}
    """
    return {"final_report": report}

# ==========================================
# 4. 組裝 Graph (照圖片結構)
# ==========================================
workflow = StateGraph(AssistantState)

# 加入節點
workflow.add_node("asr", asr_node)
workflow.add_node("minutes_taker", minutes_taker_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("writer", writer_node)

# 設定邊界 (Edges)
workflow.set_entry_point("asr")

# ASR 完成後，同時跑整理與摘要 (並行邏輯)
workflow.add_edge("asr", "minutes_taker")
workflow.add_edge("asr", "summarizer")

# 兩者都完成後匯入 Writer
workflow.add_edge("minutes_taker", "writer")
workflow.add_edge("summarizer", "writer")

workflow.add_edge("writer", END)

# 編譯
app = workflow.compile()

# ==========================================
# 5. 執行測試
# ==========================================
if __name__ == "__main__":
    # 畫出結構圖
    print(app.get_graph().draw_ascii())
    
    # 初始路徑
    initial_input = {
        "audio_path": "/home/pc-49/Desktop/nutc2504_lab/nutc2504lab_hw/HW/day3/Podcast_EP14_30s.wav"
    }
    
    print("\n--- 開始處理會議記錄 ---")
    final_state = app.invoke(initial_input)
    
    # 輸出最後結果
    print(final_state["final_report"])
    
    # 存檔
    out_dir = Path("./out")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "final_report.md", "w", encoding="utf-8") as f:
        f.write(final_state["final_report"])
    print(f"\n報告已儲存至: {out_dir}/final_report.md")