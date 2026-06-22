'''
Author: your name
Date: 2026-06-22 23:47:52
LastEditTime: 2026-06-22 23:47:52
LastEditors: limingzhangdeMac-mini.local
Description: In User Settings Edit
FilePath: /agent学习/first_agent/langchain_demo.py
'''

import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


def explain_term(topic: str) -> str:
    """Explain a frontend term in simple Chinese."""
    text = topic.lower()

    if "react" in text:
        return "React 是一个组件驱动的前端 UI 库。"

    if "vite" in text:
        return "Vite 是一个前端构建工具，启动很快。"

    return f"{topic} 是一个前端术语，你可以继续追问它的作用和场景。"


llm = ChatOpenAI(
    model="Pro/deepseek-ai/DeepSeek-V3",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)

agent = create_agent(
    model=llm,
    tools=[explain_term],
    system_prompt="你是一个前端学习助手，回答简洁清楚。",
)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "React 是什么？"}
        ]
    }
)

print(result["messages"][-1].content_blocks)