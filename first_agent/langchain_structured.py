import json
import os
from typing import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class FrontendTask(BaseModel):
    """Structured frontend learning task."""

    topic: str = Field(description="The frontend topic name")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        description="Difficulty level"
    )
    action: str = Field(description="What to do next")
    keywords: list[str] = Field(description="Important keywords")


def build_llm(api_key=None):
    resolved_api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
    if not resolved_api_key:
        raise ValueError("请先设置 SILICONFLOW_API_KEY")

    return ChatOpenAI(
        model="Pro/deepseek-ai/DeepSeek-V3",
        api_key=resolved_api_key,
        base_url="https://api.siliconflow.cn/v1",
    )


def build_prompt(question):
    return f"""
你是一个前端学习助手。

请根据用户问题返回一个 JSON 对象，不要返回 Markdown，不要返回解释文字。

JSON 字段必须固定为：
- topic: 主题名
- difficulty: easy / medium / hard
- action: 下一步学习动作
- keywords: 关键词数组

用户问题：{question}
"""


def parse_task_json(response_text):
    data = json.loads(response_text)
    return FrontendTask.model_validate(data)


def run_demo(question):
    llm = build_llm()
    prompt = build_prompt(question)
    response = llm.invoke(prompt)
    response_text = response.content if hasattr(response, "content") else response
    task = parse_task_json(response_text)

    print(task)
    print(task.topic)
    print(task.difficulty)
    print(task.action)
    print(task.keywords)
    return task


if __name__ == "__main__":
    run_demo("帮我分析一下 React hooks，给我一个适合初学者的学习任务")
