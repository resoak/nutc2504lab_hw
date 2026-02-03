import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# 1. 初始化模型
llm = ChatOpenAI(
    model="google/gemma-3-27b-it",
    temperature=0,
    base_url="https://ws-02.wade0426.me/v1",
    api_key="token"
)

# 2. 定義不同風格的 Prompt
linkedin_prompt = ChatPromptTemplate.from_template("你是一位 LinkedIn 專家，請針對主題『{topic}』寫一段專業分析。生成30個字就好")
ig_prompt = ChatPromptTemplate.from_template("你是一位 IG 網紅，請針對主題『{topic}』寫一段活潑短文。生成30個字就好")

# 3. 建立鏈並同時設定新舊參數，確保 vLLM 能正確接收
# 將 Token 限制在 30 以確保兩次執行能控制在 60 秒內
chain_limit = llm.bind(
    max_tokens=30, 
    max_completion_tokens=30
)

linkedin_chain = linkedin_prompt | chain_limit | StrOutputParser()
ig_chain = ig_prompt | chain_limit | StrOutputParser()

# 4. 使用 RunnableParallel 組合
map_chain = RunnableParallel(
    linkedin=linkedin_chain,
    instagram=ig_chain
)

# 獲取使用者輸入
user_topic = input("輸入主題：")

# --- 任務一：串流模式 (Streaming) ---
# 此階段會交錯輸出字典片段
print("\n--- 開始生成摘要 (串流模式) ---")
for chunk in map_chain.stream({"topic": user_topic}):
    print(chunk, flush=True)

print("\n" + "="*50)

# --- 任務二：批次處理 (Batch/Invoke) ---
# 此階段會紀錄處理時間
print("--- 批次處理 ---")
start_time = time.time()
result = map_chain.invoke({"topic": user_topic})
end_time = time.time()

# 5. 格式化輸出結果
print(f"耗時: {end_time - start_time:.2f} 秒")
print("-" * 50)
print(f"【LinkedIn 專家說】：\n{result['linkedin']}\n")
print("-" * 50)
print(f"【IG 網紅說】：\n{result['instagram']}")