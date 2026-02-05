import os
import json
import time
from typing import TypedDict, List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# --- 匯入參考檔案功能 (請確保檔案在同目錄) ---
try:
    from search_searxng import search_searxng
    from vlm_read_website import vlm_read_website
except ImportError as e:
    print(f"❌ 錯誤：找不到必要的工具檔案 ({e})。請確認 search_searxng.py 與 vlm_read_website.py 在同目錄下。")

# --- 1. 定義環境與模型 ---
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="", # ⚠️ 請在此填入您的 API Key
    model="google/gemma-3-27b-it",
    temperature=0.7
)

CACHE_FILE = "qa_cache.json"

# --- 2. 定義狀態 (State) ---
class State(TypedDict):
    input_query: str          # 使用者原始輸入
    knowledge_base: str       # 累積的查證資訊
    keywords: List[str]       # 生成的關鍵字
    search_links: List[dict]  # 檢索結果
    final_answer: str         # 最終回答
    is_sufficient: bool       # LLM 判斷資訊是否足夠
    loop_count: int           # 循環次數計數器

# --- 3. 定義節點 (Nodes) ---

def check_cache_node(state: State):
    """檢查快取，若命中則直接賦值 final_answer"""
    print("\n--- [節點] 快取檢查 ---")
    clean_key = state['input_query'].replace(" ", "").replace("?", "")
    
    # 預設初始化狀態
    init_state = {
        "knowledge_base": "", 
        "loop_count": 0, 
        "final_answer": "", 
        "is_sufficient": False
    }
    
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
                if clean_key in cache:
                    print(">>> ✅ Hit: 命中快取，直接跳至輸出")
                    init_state["final_answer"] = cache[clean_key]
                    return init_state
        except: pass
    
    print(">>> ❌ Miss: 無快取紀錄，進入搜尋流程")
    return init_state

def planner_node(state: State):
    """決策節點：判斷資訊是否足夠，或是否達到最大循環"""
    count = state.get('loop_count', 0)
    print(f"--- [節點] 決策 (Planner) | 當前循環次數: {count} ---")
    
    # 強制限制：若已達 3 次循環，強制視為充足，停止搜尋
    if count >= 3:
        print("⚠️ 警告：已達到最大搜尋次數 (3次)，準備彙整現有資訊。")
        return {"is_sufficient": True}

    prompt = f"""
    使用者問題: {state['input_query']}
    目前掌握資訊: {state['knowledge_base'] if state['knowledge_base'] else '目前尚未取得網頁資訊'}
    
    請評估目前資訊是否足以回答問題？
    回答要求：僅需回答 'y' (充足) 或 'n' (不足)。
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    is_y = "y" in response.content.lower()
    return {"is_sufficient": is_y}

def query_gen_node(state: State):
    """生成關鍵字節點"""
    print("--- [節點] 生成搜尋關鍵字 ---")
    prompt = f"針對問題 '{state['input_query']}'，請生成一個最適合在搜尋引擎查找的關鍵字，不要有廢話。"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"keywords": [response.content.strip()]}

def search_tool_node(state: State):
    """執行網頁檢索"""
    keyword = state['keywords'][0]
    print(f"--- [節點] 執行 SearXNG 檢索: {keyword} ---")
    results = search_searxng(keyword, limit=1) 
    return {"search_links": results}

def vlm_node(state: State):
    """VLM 視覺處理節點，並累加計數器"""
    print("--- [節點] VLM 視覺閱讀網頁 ---")
    current_count = state.get("loop_count", 0)
    
    if not state.get('search_links'):
        return {"knowledge_base": state['knowledge_base'] + "\n未找到相關連結。", "loop_count": current_count + 1}
    
    target = state['search_links'][0]
    # 調用 Playwright 截圖與 VLM 分析
    summary = vlm_read_website(target['url'], target['title'])
    
    new_kb = f"{state['knowledge_base']}\n\n[來源: {target['url']}]\n{summary}"
    return {"knowledge_base": new_kb, "loop_count": current_count + 1}

def output_node(state: State):
    """彙整資訊產生最終回答，並存入快取"""
    print("--- [節點] 產生最終輸出 ---")
    
    # 如果是快取命中的，直接結束
    if state.get("final_answer"):
        return {"final_answer": state["final_answer"]}

    prompt = f"請根據以下查證資訊，完整且精確地回答使用者問題 '{state['input_query']}':\n{state['knowledge_base']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # 寫入快取
    clean_key = state['input_query'].replace(" ", "").replace("?", "")
    cache_data = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f: cache_data = json.load(f)
        except: pass
    
    cache_data[clean_key] = response.content
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=4)
        
    return {"final_answer": response.content}

# --- 4. 構建圖表 (Graph) ---

def cache_router(state: State):
    return "hit" if state.get("final_answer") else "miss"

def planner_router(state: State):
    # 如果循環次數 >= 3，無論 LLM 判斷為何，一律去 output
    if state.get("loop_count", 0) >= 3:
        return "y"
    return "y" if state.get("is_sufficient") else "n"

workflow = StateGraph(State)

workflow.add_node("check_cache", check_cache_node)
workflow.add_node("planner", planner_node)
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("search_tool", search_tool_node)
workflow.add_node("vlm_process", vlm_node)
workflow.add_node("output", output_node)

workflow.set_entry_point("check_cache")

# 設定連線邏輯
workflow.add_conditional_edges("check_cache", cache_router, {"hit": "output", "miss": "planner"})
workflow.add_conditional_edges("planner", planner_router, {"y": "output", "n": "query_gen"})

workflow.add_edge("query_gen", "search_tool")
workflow.add_edge("search_tool", "vlm_process")
workflow.add_edge("vlm_process", "planner") # 循環點
workflow.add_edge("output", END)

app = workflow.compile()

# --- 5. 執行區 (互動介面版) ---
if __name__ == "__main__":
    print("\n" + "========================================")
    print("  🚀 自動查證 AI 系統 (Gemma-3 VLM) 已啟動")
    print("  (輸入 'exit' 或 '離開' 結束程式)")
    print("========================================\n")

    while True:
        query = input("❓ 請輸入您想查詢的問題：").strip()

        if query.lower() in ['exit', 'quit', '離開', '退出']:
            print("👋 程式已安全退出，再見！")
            break
        
        if not query:
            continue

        print(f"\n⚙️ 正在處理請求，請稍候...")
        
        try:
            # 啟動 LangGraph
            final_state = app.invoke({
                "input_query": query, 
                "knowledge_base": "", 
                "loop_count": 0, 
                "final_answer": ""
            })
            
            print("\n" + "✨" + "—"*48)
            print(f"【最終回答】\n\n{final_state['final_answer']}")
            print("—"*50 + "\n")
            
        except Exception as e:
            print(f"❌ 發生未預期錯誤: {e}")