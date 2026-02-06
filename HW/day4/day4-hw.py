import os
import json
import base64
import requests
from typing import TypedDict, List, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from playwright.sync_api import sync_playwright

# --- 1. 環境設定 ---
SEARXNG_URL = "https://puli-8080.huannago.com/search"
CACHE_FILE = "qa_cache.json"

llm = ChatOpenAI(
    base_url="https://ws-05.huannago.com/v1",
    api_key="YOUR_API_KEY", 
    model="google/gemma-3-27b-it",
    temperature=0 
)

# --- 2. 狀態定義 ---
class State(TypedDict):
    input_query: str
    knowledge_base: str
    keywords: List[str]
    search_links: List[dict]
    visited_urls: Annotated[List[str], operator.add]
    final_answer: str
    is_sufficient: bool 
    loop_count: int

# --- 3. 核心工具 ---
def internal_vlm_read_website(url: str, original_query: str) -> str:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_viewport_size({"width": 1280, "height": 900})
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(3000)
            b64 = base64.b64encode(page.screenshot()).decode('utf-8')
            browser.close()
            
            msg = [
                {"type": "text", "text": f"""你是一位事實分析官。請針對用戶的問題「{original_query}」分析此網頁截圖：
                1. **來源性質**：該網頁是否為官方發布、權威報導或一般社群討論？
                2. **核心事實**：提取所有與問題相關的時間、數據或事件狀態。
                3. **變化記錄**：若問題涉及變動，請精確記錄「變動前」與「變動後」的具體內容。
                4. **可信度**：內容中是否有標註『傳聞』、『猜測』或『非官方證實』等字眼？"""},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
            res = llm.invoke([HumanMessage(content=msg)])
            return res.content
    except Exception as e: return f"讀取錯誤: {e}"

# --- 4. 流程節點實現 (完全不含標的資訊) ---

def check_cache_node(state: State):
    print(f"🔎 [步驟 1] 檢查快取...")
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
            if state['input_query'] in cache:
                print("🎯 [命中快取]")
                return {"final_answer": cache[state['input_query']]}
    return {"knowledge_base": "", "loop_count": 0, "visited_urls": []}

def planner_node(state: State):
    if state['loop_count'] >= 5: return {"is_sufficient": True}

    print(f"🧠 [步驟 2] 決策評估 (第 {state['loop_count']} 輪)...")
    prompt = f"""用戶問題：{state['input_query']}
    當前收集資訊：{state['knowledge_base']}
    
    請判斷：
    1. 資訊是否包含來自官方主體（相關公司/機構）的直接證據？
    2. 如果問題涉及次數計算，是否有明確的變化歷程記錄？
    3. 是否有足夠證據排除第三方媒體的猜測？
    
    資訊是否充裕？請回答 y 或 n。"""
    
    res = llm.invoke([HumanMessage(content=prompt)])
    return {"is_sufficient": 'y' in res.content.lower()}

def query_gen_node(state: State):
    print("\n💡 [步驟 3] 動態生成與修正搜尋策略...")
    
    # 提取先前的搜尋歷史與目前的知識儲備
    past_keywords = state.get('keywords', [])
    current_kb = state.get('knowledge_base', '目前尚無有效資訊')
    
    # 建構引導 Prompt，讓 LLM 具備「反思」能力
    history_str = f"已嘗試過的關鍵字：{', '.join(past_keywords)}" if past_keywords else "這是第一次搜尋。"
    
    prompt = f"""你是一位專業的情報分析官。
    【用戶問題】：{state['input_query']}
    【搜尋歷史】：{history_str}
    【目前掌握資訊簡述】：{current_kb[:500]}... (略)

    請執行以下思考：
    1. 檢視目前資訊是否已足以回答問題？
    2. 如果不足，是因為「找不到官網」、「資訊太舊」還是「缺乏具體數據」？
    3. 避開已嘗試過的關鍵字，生成一個更高精確度、英文為主的關鍵字。
    
    請直接回傳關鍵字字串，不要解釋說明。"""
    
    res = llm.invoke([HumanMessage(content=prompt)])
    new_kw = res.content.strip().replace('"', '').replace('*', '')
    
    # --- 關鍵修改：Print 出生成的關鍵字以利觀測 ---
    print(f"🔄 策略修正中...")
    print(f"   ↳ 原始問題：{state['input_query']}")
    if past_keywords:
        print(f"   ↳ 歷史關鍵字：{past_keywords}")
    print(f"   ↳ ✨ 新生成的優化關鍵字：【 {new_kw} 】")
    
    return {"keywords": state.get('keywords', []) + [new_kw]}

def search_node(state: State):
    print(f"📡 [步驟 4] 檢索網路資源...")
    try:
        r = requests.get(SEARXNG_URL, params={"q": state['keywords'][-1], "format": "json"}, timeout=15).json()
        return {"search_links": r.get('results', [])[:3]}
    except: return {"search_links": []}

def vlm_and_value_node(state: State):
    print("📸 [步驟 5] VLM 事實提取...")
    links = state.get('search_links', [])
    new_info = ""
    for link in links:
        if link['url'] in state['visited_urls']: continue
        print(f"📖 閱讀來源：{link['url'][:50]}...")
        info = internal_vlm_read_website(link['url'], state['input_query'])
        new_info += f"\n[來源: {link['url']}]\n{info}\n"
        break 
    return {"knowledge_base": state['knowledge_base'] + new_info, "visited_urls": [link['url']], "loop_count": state['loop_count'] + 1}

def output_node(state: State):
    if state.get("final_answer"): return state
    print("🏁 [步驟 6] 彙整最終事實報告...")
    prompt = f"""請針對用戶問題「{state['input_query']}」產出查證報告。
    
    【規則】：
    1. **證據分級**：優先採用官方主體的直接證據，排除未經證實的傳聞。
    2. **變化核對**：如果涉及變動次數，請列出具體的時間軸節點。
    3. **誠實性**：若證據不足，請如實說明哪些部分屬於官方確定，哪些屬於媒體推測。
    
    筆記內容：
    {state['knowledge_base']}"""
    
    res = llm.invoke([HumanMessage(content=prompt)])
    final_ans = res.content
    
    # 快取處理
    try:
        if not os.path.exists(CACHE_FILE): cache = {}
        else:
            with open(CACHE_FILE, "r", encoding="utf-8") as f: cache = json.load(f)
        cache[state['input_query']] = final_ans
        with open(CACHE_FILE, "w", encoding="utf-8") as f: json.dump(cache, f, ensure_ascii=False, indent=4)
    except: pass
    
    return {"final_answer": final_ans}

# --- 5. 構建圖表 ---



workflow = StateGraph(State)
workflow.add_node("check_cache", check_cache_node)
workflow.add_node("planner", planner_node)
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("search_tool", search_node)
workflow.add_node("vlm_process", vlm_and_value_node)
workflow.add_node("output", output_node)

workflow.set_entry_point("check_cache")
workflow.add_conditional_edges("check_cache", lambda x: "hit" if x.get("final_answer") else "miss", {"hit": "output", "miss": "planner"})
workflow.add_conditional_edges("planner", lambda x: "y" if x["is_sufficient"] else "n", {"y": "output", "n": "query_gen"})
workflow.add_edge("query_gen", "search_tool")
workflow.add_edge("search_tool", "vlm_process")
workflow.add_edge("vlm_process", "planner")
workflow.add_edge("output", END)

app = workflow.compile()

# --- 6. 執行介面 ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🕵️ 通用型自律事實查證引擎 (標的去中心化版)")
    try: app.get_graph().print_ascii()
    except: pass
    print("="*50)

    while True:
        user_input = input("\n🔎 請輸入要查證的問題 (exit 退出)：").strip()
        if not user_input or user_input.lower() == 'exit': break
        
        # 執行並顯示過程
        result = app.invoke({
            "input_query": user_input, 
            "knowledge_base": "", 
            "keywords": [], 
            "loop_count": 0, 
            "final_answer": "", 
            "visited_urls": []
        })
        
        print("\n" + "★"*25)
        print("✨ 【 查 證 報 告 】")
        print(result['final_answer'])
        print(f"📊 調查深度：{result['loop_count']} 輪")
        print("★"*25)