'''
Author: your name
Date: 2026-06-23 23:12:41
LastEditTime: 2026-06-23 23:12:53
LastEditors: limingzhangdeMac-mini.local
Description: In User Settings Edit
FilePath: /agent学习/first_agent/langchain_chat.py
'''
import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver


@tool
def explain_term(topic: str) -> str:
    """Explain a frontend term in simple Chinese."""
    text = topic.lower()

    if "react" in text:
        return "React 是一个组件驱动的前端 UI 库，适合构建交互界面。"

    if "vite" in text:
        return "Vite 是一个前端构建工具，开发启动快，热更新也快。"

    if "flex" in text:
        return "Flex 是一维布局，适合处理一行或一列的排列。"

    return f"{topic} 是一个前端相关术语，你可以继续追问它的作用、场景和对比。"


llm = ChatOpenAI(
    model="Pro/deepseek-ai/DeepSeek-V3",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=[explain_term],
    system_prompt="你是一个前端学习助手，回答简洁、清楚、像在带初学者。",
    checkpointer=checkpointer,
)

thread_id = "frontend-study-chat"

print("前端术语助手已启动，输入 quit 退出。")

while True:
    question = input("\n你：").strip()

    if question.lower() in {"quit", "exit"}:
        print("助手：下次继续。")
        break

    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id}},
    )

    print("助手：", result["messages"][-1].text())