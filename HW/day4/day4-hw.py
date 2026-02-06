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

# 使用 Gemma-3-27b 進行推理
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
    """透過 Playwright 截圖並讓 VLM 分析網頁事實"""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_viewport_size({"width": 1280, "height": 900})
            # 等待網路閒置，確保圖片與表格載入
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(2000)
            b64 = base64.b64encode(page.screenshot(full_page=False)).decode('utf-8')
            browser.close()
            
            msg = [
                {"type": "text", "text": f"""你是一位事實分析官。請針對問題「{original_query}」分析網頁截圖：
                1. **核心事實**：提取與問題直接相關的時間、數據、聲明原文。
                2. **來源信度**：判斷此為官方公告、媒體報導或個人評論。
                3. **細節抓取**：若有表格或細則，請精確列出。
                如果網頁內容完全無關，請回覆「無相關資訊」。"""},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
            res = llm.invoke([HumanMessage(content=msg)])
            return res.content
    except Exception as e: 
        return f"讀取錯誤: {str(e)}"

# --- 4. 流程節點實現 ---

def check_cache_node(state: State):
    print(f"🔎 [步驟 1] 檢查快取...")
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
            if state['input_query'] in cache:
                print("🎯 [命中快取] 載入現有查證報告。")
                return {"final_answer": cache[state['input_query']]}
    return {"knowledge_base": "", "loop_count": 0, "visited_urls": []}

def planner_node(state: State):
    if state['loop_count'] >= 5: 
        print("🚨 [系統] 已達最大輪次上限，強制結束調查。")
        return {"is_sufficient": True}

    print(f"\n🕵️ [步驟 2] 指揮官 (Planner) 評估 (輪次 {state['loop_count']})...")
    
    prompt = f"""
    你是一位嚴謹的事實審核指揮官。請根據目前掌握的證據評估是否已足夠回答用戶問題。
    
    【用戶問題】：{state['input_query']}
    【目前證據庫】：{state['knowledge_base'][-2000:]}

    判斷準則：
    1. 關鍵事實（如日期、數值、動作主體）是否明確？
    2. 是否有官方來源背書？
    3. 多個網頁間是否有資訊衝突？

    若已足夠，回傳 [COMPLETE]。
    若需更多細節，回傳 [CONTINUE] 並具體描述「還缺什麼資訊」。
    """
    
    res = llm.invoke([HumanMessage(content=prompt)])
    content = res.content.upper()
    is_sufficient = "[COMPLETE]" in content

    print(f"📝 評估報告：{'✅ 資訊已完整' if is_sufficient else f'❌ 需繼續調查。缺口：{content}'}")
    return {"is_sufficient": is_sufficient}

def query_gen_node(state: State):
    print(f"\n💡 [步驟 3] 策略官 (Query Gen) 正在分析與修正策略...")
    
    past_kws = state.get('keywords', [])
    knowledge = state.get('knowledge_base', '')

    prompt = f"""
    你是一位情報搜尋專家。請生成一個精準的英文搜尋詞，以填補目前查證的資訊缺口。
    
    【用戶問題】：{state['input_query']}
    【已試過關鍵字】：{", ".join(past_kws) if past_kws else "無"}
    【查證進度摘要】：{knowledge[:400] if knowledge else "尚未獲得有效資訊"}

    規則：
    1. 避免重複已使用的詞。
    2. 生成更具體、朝向「官方公告」或「原始文件」的搜尋詞。
    3. 僅回傳關鍵字字串，不需解釋。
    """
    
    res = llm.invoke([HumanMessage(content=prompt)])
    new_kw = res.content.strip().replace('"', '').replace('*', '')
    
    print(f"🚀 新生成的精準關鍵字：【 {new_kw} 】")
    return {"keywords": state.get('keywords', []) + [new_kw]}

def search_node(state: State):
    current_kw = state['keywords'][-1]
    print(f"📡 [步驟 4] 檢索中：{current_kw}...")
    try:
        r = requests.get(SEARXNG_URL, params={"q": current_kw, "format": "json"}, timeout=15).json()
        raw_results = r.get('results', [])
        # 過濾已造訪網址
        filtered = [res for res in raw_results if res['url'] not in state['visited_urls']]
        return {"search_links": filtered[:3]} # 提取前 3 篇供 VLM 閱讀
    except:
        return {"search_links": []}

def vlm_and_value_node(state: State):
    links = state.get('search_links', [])
    print(f"📸 [步驟 5] VLM 事實提取 (預計掃描 {len(links)} 篇文章)...")
    
    new_info_batch = ""
    newly_visited = []
    
    for i, link in enumerate(links):
        url = link['url']
        print(f"📖 [{i+1}/{len(links)}] 正在視覺化掃描：{url[:50]}...")
        
        info = internal_vlm_read_website(url, state['input_query'])
        
        if "無相關資訊" not in info:
            new_info_batch += f"\n[來源 {state['loop_count']+1}-{i+1}]: {url}\n{info}\n"
        
        newly_visited.append(url)
        
    return {
        "knowledge_base": state['knowledge_base'] + new_info_batch, 
        "visited_urls": newly_visited, 
        "loop_count": state['loop_count'] + 1
    }

def output_node(state: State):
    if state.get("final_answer"): return state
    print("\n🏁 [步驟 6] 彙整最終查證報告...")
    
    prompt = f"""針對問題「{state['input_query']}」，請根據以下調查事實撰寫一份嚴謹的報告。
    
    【事實庫】：
    {state['knowledge_base']}

    格式要求：
    1. **結論先行**：直接回答查證結果。
    2. **證據列表**：列出支持事實的官方來源與具體數據。
    3. **爭議點/缺失**：若有矛盾或查不到的部分，請如實說明。
    """
    
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
    print("\n🕵️ 深度事實查證引擎啟動...")
    while True:
        print(app.get_graph().print_ascii())
        user_input = input("\n🔎 請輸入要查證的問題 (exit 退出)：").strip()
        if not user_input or user_input.lower() == 'exit': break
        
        result = app.invoke({
            "input_query": user_input, 
            "knowledge_base": "", 
            "keywords": [], 
            "loop_count": 0, 
            "final_answer": "", 
            "visited_urls": []
        })
        
        print("\n" + "★"*35)
        print("✨ 【 最終查證報告 】")
        print(result['final_answer'])
        print(f"📊 調查統計：歷經 {result['loop_count']} 輪調查，共掃描 {len(result['visited_urls'])} 個網頁")
        print("★"*35)