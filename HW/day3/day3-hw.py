import time
import requests
import json
from pathlib import Path
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# ==========================================
# 1. 基礎設定 (請務必填入你的 TOKEN)
# ==========================================
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",  # <--- 在此填入 Token
    model="google/gemma-3-27b-it",
    temperature=0
)

# ==========================================
# 2. ASR 核心函數
# ==========================================
def run_asr_task(wav_path: str):
    BASE = "https://3090api.huannago.com"
    CREATE_URL = f"{BASE}/api/v1/subtitle/tasks"
    auth = ("nutc2504", "nutc2504")

    with open(wav_path, "rb") as f:
        r = requests.post(CREATE_URL, files={"audio": f}, timeout=60, auth=auth)
    r.raise_for_status()
    task_id = r.json()["id"]
    
    txt_url = f"{BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type=TXT" 
    srt_url = f"{BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type=SRT"

    def wait_download(url: str):
        for _ in range(300): 
            resp = requests.get(url, auth=auth)
            if resp.status_code == 200: return resp.text
            time.sleep(2)
        return None

    print(f"\n--- 📡 ASR 任務 {task_id} 啟動 ---")
    txt_content = wait_download(txt_url)
    srt_content = wait_download(srt_url)
    
    # 💡 在這裡增加 Print 出原始轉錄內容
    print("\n[原始 TXT 內容]:")
    print(txt_content)
    print("\n[原始 SRT 內容]:")
    print(srt_content)
    
    return txt_content, srt_content

# ==========================================
# 3. 定義 LangGraph 狀態與節點
# ==========================================
class AssistantState(TypedDict):
    audio_path: str
    txt_content: str  
    srt_content: str  
    minutes: str      
    summary: str      
    final_report: str 

def asr_node(state: AssistantState):
    txt, srt = run_asr_task(state["audio_path"])
    return {"txt_content": txt, "srt_content": srt}

def minutes_taker_node(state: AssistantState):
    print("\n-> 正在產出詳細逐字稿...")
    prompt = f"""請根據以下 SRT 內容，整理出『詳細逐字稿』。
格式要求：
## 🎙️ 詳細記錄 (Detailed Minutes)
**時間** | **發言內容**
--- | ---
{state['srt_content']}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"minutes": response.content}

def summarizer_node(state: AssistantState):
    print("-> 正在產出重點摘要...")
    prompt = f"""請針對以下會議內容整理出『重點摘要』。
格式要求：
# 📄 智慧會議記錄報告
## 🎯 重點摘要 (Executive Summary)
**決策結果：** [填寫結果]
**待辦事項：** [填寫清單]
內容：{state['txt_content']}"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"summary": response.content}

def writer_node(state: AssistantState):
    report = f"{state['summary']}\n\n---\n\n{state['minutes']}"
    return {"final_report": report}

# ==========================================
# 4. 組裝 Graph
# ==========================================
workflow = StateGraph(AssistantState)
workflow.add_node("asr", asr_node)
workflow.add_node("minutes_taker", minutes_taker_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("writer", writer_node)

workflow.set_entry_point("asr")
workflow.add_edge("asr", "minutes_taker")
workflow.add_edge("asr", "summarizer")
workflow.add_edge("minutes_taker", "writer")
workflow.add_edge("summarizer", "writer")
workflow.add_edge("writer", END)
app = workflow.compile()

# ==========================================
# 5. 執行並產出結果
# ==========================================
if __name__ == "__main__":
    initial_input = {"audio_path": "/home/pc-49/Downloads/Podcast_EP14_30s.wav"}
    
    print("\n--- 🚀 開始智慧助理流程 ---")
    result = app.invoke(initial_input)
    
    # --- 儲存檔案 ---
    out_dir = Path("./out")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "transcription.txt").write_text(result["txt_content"], encoding="utf-8")
    (out_dir / "transcription.srt").write_text(result["srt_content"], encoding="utf-8")
    (out_dir / "out.md").write_text(result["final_report"], encoding="utf-8")

    # --- 💡 列印最終報告內容 ---
    print("\n" + "="*30 + " 最終報告 (out.md) " + "="*30)
    print(result["final_report"])
    print("="*75)
    print(f"\n✅ 任務完成！所有內容已列印並存檔至 {out_dir} 資料夾。")