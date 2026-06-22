import json
import os
from datetime import datetime
from pathlib import Path


def decide_next_step(goal, skills):
    normalized_skills = {skill.lower() for skill in skills}

    if goal == "agent" and "python" in normalized_skills and "api" in normalized_skills:
        return {
            "action": "start_project",
            "message": "可以开始做一个可展示的 agent 项目。",
        }

    if goal == "agent" and "python" in normalized_skills:
        return {
            "action": "learn_api",
            "message": "你已经有 Python 基础，下一步补 API，再做 agent。",
        }

    return {
        "action": "learn_python",
        "message": "先补 Python 最小必备，再进入 agent。",
    }


TOOL_SCHEMAS = [
    {
        "name": "learn_python",
        "description": "用户还缺 Python 基础时使用。",
    },
    {
        "name": "learn_api",
        "description": "用户会 Python，但还需要补 API、HTTP、JSON 基础时使用。",
    },
    {
        "name": "start_project",
        "description": "用户已经具备 Python 和 API 基础，可以开始做 agent 项目时使用。",
    },
    {
        "name": "plan_learning",
        "description": "用户想先系统学习 agent，再决定做项目时使用。",
    },
    {
        "name": "clarify_goal",
        "description": "用户目标或技能描述不够明确，需要先澄清时使用。",
    },
]


def build_tool_instructions(tool_schemas):
    lines = ["可用 action 只能从下面工具中选择："]
    for tool_schema in tool_schemas:
        lines.append(f"- {tool_schema['name']}：{tool_schema['description']}")
    return "\n".join(lines)


def allowed_actions():
    return {tool_schema["name"] for tool_schema in TOOL_SCHEMAS}


def validate_decision(decision):
    action = decision.get("action")
    if action not in allowed_actions():
        raise ValueError(f"未知 action：{action}")

    args = decision.setdefault("args", {})
    if not isinstance(args, dict):
        raise ValueError("args 必须是对象。")

    if "message" not in decision:
        decision["message"] = ""

    return decision


def build_prompt(goal, skills, memory=None):
    skills_text = ", ".join(skills)
    tool_instructions = build_tool_instructions(TOOL_SCHEMAS)
    memory = memory or {}
    clarification_lines = []
    for item in memory.get("clarifications", []):
        clarification_lines.append(
            f"- 问题：{item.get('question', '')}\n  回答：{item.get('answer', '')}"
        )
    clarification_text = "\n".join(clarification_lines) or "无"
    return f"""
你是一个 agent 学习顾问。

用户目标：{goal}
用户技能：{skills_text}
澄清记录：
{clarification_text}

请判断用户下一步应该做什么。
{tool_instructions}

只返回 JSON，不要返回 Markdown。
JSON 格式：
{{"action": "动作代号", "message": "给用户看的中文建议", "args": {{"focus": "可选关注点"}}}}
"""


def parse_llm_response(response_text):
    candidate_texts = [response_text]

    stripped_text = response_text.strip()
    if stripped_text.startswith("```"):
        lines = stripped_text.splitlines()
        if len(lines) >= 3:
            candidate_texts.append("\n".join(lines[1:-1]))

    json_start = response_text.find("{")
    json_end = response_text.rfind("}")
    if json_start != -1 and json_end != -1 and json_end > json_start:
        candidate_texts.append(response_text[json_start:json_end + 1])

    last_error = None
    for candidate_text in candidate_texts:
        try:
            result = json.loads(candidate_text)
            return validate_decision(result)
        except json.JSONDecodeError as error:
            last_error = error

    preview = response_text[:120] if response_text else "<empty>"
    raise ValueError(
        "模型没有返回合法 JSON。"
        f"原始返回开头：{preview}"
    ) from last_error


def decide_next_step_with_llm(goal, skills, model, memory=None):
    prompt = build_prompt(goal, skills, memory=memory)
    response_text = model(prompt)
    return parse_llm_response(response_text)


def load_course_catalog():
    sidebar_path = Path(__file__).resolve().parent.parent / "hello-agents" / "docs" / "_sidebar.md"
    if not sidebar_path.exists():
        return []

    chapters = []
    for line in sidebar_path.read_text(encoding="utf-8").splitlines():
        stripped_line = line.strip()
        if stripped_line.startswith("- [第") and "](" in stripped_line:
            title = stripped_line.split("[", 1)[1].split("]", 1)[0]
            chapters.append(title)

    return chapters


def docs_root():
    return Path(__file__).resolve().parent.parent / "hello-agents" / "docs"


def find_chapter_file(chapter_title):
    for path in docs_root().glob("chapter*/*.md"):
        if path.stem == chapter_title:
            return path
    return None


def read_chapter_excerpt(chapter_title, max_lines=6):
    path = find_chapter_file(chapter_title)
    if not path:
        return f"未找到章节文件：{chapter_title}"

    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped_line = line.strip()
        if stripped_line:
            lines.append(stripped_line)
        if len(lines) >= max_lines:
            break

    return "\n".join(lines)


def recommend_course_chapters(decision):
    chapters = load_course_catalog()
    action = decision.get("action")

    chapter_map = {
        "learn_python": ["第一章 初识智能体", "第三章 大语言模型基础"],
        "learn_api": ["第六章 框架开发实践", "第七章 构建你的Agent框架"],
        "start_project": [
            "第六章 框架开发实践",
            "第七章 构建你的Agent框架",
            "第十三章 智能旅行助手",
            "第十六章 毕业设计",
        ],
        "plan_learning": [
            "第三章 大语言模型基础",
            "第六章 框架开发实践",
            "第七章 构建你的Agent框架",
        ],
    }
    selected_titles = chapter_map.get(action, [])
    selected_chapters = [chapter for chapter in chapters if chapter in selected_titles]

    if action == "plan_learning":
        focus = decision.get("memory", {}).get("preferences", {}).get("focus")
        focus_text = "前端 Agent" if focus == "frontend" else "Agent"
        skills = decision.get("skills", [])
        has_python = any(skill.lower() == "python" for skill in skills)
        step_one = "补 Python + API 基础" if not has_python else "补 API + JSON + HTTP"
        step_one_chapter = "第三章 大语言模型基础"
        step_two_chapter = "第六章 框架开发实践"
        step_three_chapter = "第七章 构建你的Agent框架"

        return (
            f"给你一份每天 1 小时的 3 步学习计划，适合先系统学习 {focus_text}：\n"
            f"1. 第 1 周：{step_one}，先读《{step_one_chapter}》。\n"
            f"2. 第 2 周：理解 Agent 怎么接模型和工具，重点读《{step_two_chapter}》。\n"
            f"3. 第 3 周：自己拼一个最小可运行 Agent，重点读《{step_three_chapter}》。"
        )

    if not selected_chapters:
        return "先学 HTTP、JSON、requests/httpx，再把 API 调用接进 agent。"

    focus = decision.get("args", {}).get("focus")
    if not focus:
        focus = decision.get("memory", {}).get("preferences", {}).get("focus")
    focus_text = ""
    if focus == "backend":
        focus_text = "重点关注后端/API 接入。"
    elif focus == "frontend":
        focus_text = "重点关注前端 Agent 接入。"

    excerpt = read_chapter_excerpt(selected_chapters[0], max_lines=3)
    return (
        "推荐你读本地 hello-agents 教程："
        + "；".join(selected_chapters)
        + focus_text
        + "\n内容摘录："
        + excerpt
    )


def suggest_python_resources(decision):
    return recommend_course_chapters(decision)


def suggest_api_resources(decision):
    return recommend_course_chapters(decision)


def suggest_project_resources(decision):
    return recommend_course_chapters(decision)


def suggest_learning_plan(decision):
    return recommend_course_chapters(decision)


def suggest_clarify_goal(_decision):
    return "你可以继续说清楚：你想做求职作品集、接单项目，还是先系统学习 agent 原理？"


def extract_preferences(text):
    preferences = {}
    lowered_text = text.lower()

    if "作品集" in text:
        preferences["goal_type"] = "portfolio"
    elif "接单" in text:
        preferences["goal_type"] = "freelance"
    elif "学习" in text or "原理" in text:
        preferences["goal_type"] = "study"

    if "前端" in text or "frontend" in lowered_text:
        preferences["focus"] = "frontend"
    elif "后端" in text or "backend" in lowered_text:
        preferences["focus"] = "backend"

    return preferences


TOOLS = {
    "learn_python": suggest_python_resources,
    "learn_api": suggest_api_resources,
    "start_project": suggest_project_resources,
    "plan_learning": suggest_learning_plan,
    "clarify_goal": suggest_clarify_goal,
}


def execute_action(decision):
    action = decision.get("action")
    tool = TOOLS.get(action)

    if not tool:
        return {
            **decision,
            "tool_result": f"动作 {action} 还没有工具，先只展示模型建议。",
        }

    return {
        **decision,
        "tool_result": tool(decision),
    }


def run_agent(goal, skills, model, memory=None):
    decision = decide_next_step_with_llm(goal, skills, model, memory=memory)
    decision["memory"] = memory or {}
    decision["skills"] = skills
    return execute_action(decision)


def build_final_prompt(goal, skills, observation):
    skills_text = ", ".join(skills)
    preferences = observation.get("memory", {}).get("preferences", {})
    preference_parts = []
    if preferences.get("goal_type") == "portfolio":
        preference_parts.append("作品集导向")
    elif preferences.get("goal_type") == "freelance":
        preference_parts.append("接单导向")
    elif preferences.get("goal_type") == "study":
        preference_parts.append("学习导向")

    if preferences.get("focus") == "frontend":
        preference_parts.append("前端方向")
    elif preferences.get("focus") == "backend":
        preference_parts.append("后端方向")

    preference_text = "、".join(preference_parts) if preference_parts else "无"
    return f"""
你是一个 agent 学习顾问。

用户目标：{goal}
用户技能：{skills_text}
用户偏好：{preference_text}

你刚才的判断：{observation.get("message", "")}
工具执行结果：{observation.get("tool_result", "")}

请基于工具执行结果，给用户一个简短、自然、可执行的最终回答。
不要返回 JSON，直接返回最终回答。
"""


def run_agent_loop(goal, skills, model, memory=None):
    observation = run_agent(goal, skills, model, memory=memory)
    final_prompt = build_final_prompt(goal, skills, observation)
    final_answer = model(final_prompt)

    return {
        **observation,
        "final_answer": final_answer,
        "trace": [
            {
                "step": "decision",
                "action": observation.get("action"),
                "message": observation.get("message"),
                "args": observation.get("args", {}),
            },
            {
                "step": "tool",
                "action": observation.get("action"),
                "tool_result": observation.get("tool_result"),
            },
            {
                "step": "final",
                "final_answer": final_answer,
            },
        ],
    }


def run_agent_conversation(goal, skills, model, ask_user=input):
    memory = {"clarifications": [], "preferences": {}}
    first_observation = run_agent(goal, skills, model, memory=memory)
    if first_observation.get("action") != "clarify_goal":
        final_prompt = build_final_prompt(goal, skills, first_observation)
        final_answer = model(final_prompt)
        return {
            **first_observation,
            "final_answer": final_answer,
            "memory": memory,
            "trace": [
                {
                    "step": "decision",
                    "action": first_observation.get("action"),
                    "message": first_observation.get("message"),
                    "args": first_observation.get("args", {}),
                },
                {
                    "step": "tool",
                    "action": first_observation.get("action"),
                    "tool_result": first_observation.get("tool_result"),
                },
                {
                    "step": "final",
                    "final_answer": final_answer,
                },
            ],
        }

    clarification_answer = ask_user(first_observation["tool_result"] + "\n你的补充：")
    memory["clarifications"].append({
        "question": first_observation.get("tool_result"),
        "answer": clarification_answer,
    })
    memory["preferences"].update(extract_preferences(clarification_answer))
    second_result = run_agent_loop(goal, skills, model, memory=memory)
    second_result["memory"] = memory
    second_result["trace"] = [
        {
            "step": "clarification",
            "question": first_observation.get("tool_result"),
            "answer": clarification_answer,
        },
        *second_result["trace"],
    ]
    return second_result


def format_trace(trace):
    lines = ["TRACE"]
    for step in trace:
        step_name = step.get("step")
        if step_name == "decision":
            lines.append(
                f"[decision] action={step.get('action')} "
                f"args={step.get('args', {})} message={step.get('message', '')}"
            )
        elif step_name == "tool":
            lines.append(
                f"[tool] action={step.get('action')} "
                f"result={step.get('tool_result', '')}"
            )
        elif step_name == "final":
            lines.append(f"[final] {step.get('final_answer', '')}")
        else:
            lines.append(f"[{step_name}] {step}")

    return "\n".join(lines)


def save_run_log(profile, result, runs_dir=None):
    resolved_runs_dir = runs_dir or Path(__file__).resolve().parent / "runs"
    resolved_runs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = resolved_runs_dir / f"run-{timestamp}.json"
    payload = {
        "profile": profile,
        "final_answer": result.get("final_answer"),
        "memory": result.get("memory", {}),
        "trace": result.get("trace", []),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def validate_api_key(name, value):
    if not value:
        return

    try:
        value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(
            f"{name} 不是有效的 API key。请不要填中文占位文本，"
            f"应该设置成控制台生成的 sk-... 字符串。"
        ) from error


def create_openai_model(model_name="gpt-4.1-mini"):
    from openai import OpenAI

    validate_api_key("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    client = OpenAI()

    def model(prompt):
        response = client.responses.create(
            model=model_name,
            input=prompt,
        )
        return response.output_text

    return model


def create_siliconflow_model(api_key, model_name="Pro/deepseek-ai/DeepSeek-V3"):
    from openai import OpenAI

    validate_api_key("SILICONFLOW_API_KEY", api_key)
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.siliconflow.cn/v1",
    )

    def model(prompt):
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return response.choices[0].message.content

    return model


def create_rule_based_model(goal, skills):
    def model(prompt):
        if "最终回答" in prompt:
            if "作品集" in prompt and "前端" in prompt:
                return "最终建议：先读第六章和第七章，把它整理成一个前端 Agent 作品集小项目。"
            return "最终建议：先按工具推荐读对应章节，再做一个最小 agent 小练习。"

        if "系统学习" in prompt or "agent 原理" in prompt:
            return json.dumps(
                {
                    "action": "plan_learning",
                    "message": "先按学习路线系统补齐，再开始做 agent。",
                    "args": {},
                },
                ensure_ascii=False,
            )

        return json.dumps(
            decide_next_step(goal, skills),
            ensure_ascii=False,
        )

    return model


def pick_model(goal, skills, siliconflow_api_key=None, openai_api_key=None):
    resolved_siliconflow_api_key = siliconflow_api_key or os.getenv("SILICONFLOW_API_KEY")
    resolved_openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

    if resolved_siliconflow_api_key:
        return create_siliconflow_model(api_key=resolved_siliconflow_api_key)

    if resolved_openai_api_key:
        return create_openai_model()

    return create_rule_based_model(goal, skills)


def parse_skills(raw_skills):
    return [
        skill.strip()
        for skill in raw_skills.split(",")
        if skill.strip()
    ]


def build_profile():
    goal = input("你的目标是什么？比如 agent：").strip()
    raw_skills = input("你会哪些技能？用英文逗号分隔，比如 frontend, python：")

    return {
        "goal": goal,
        "skills": parse_skills(raw_skills),
    }


def run():
    profile = build_profile()

    try:
        model = pick_model(profile["goal"], profile["skills"])
        result = run_agent_conversation(
            profile["goal"],
            profile["skills"],
            model,
        )
        print(result["final_answer"])
        save_run_log(profile, result)
        if os.getenv("SHOW_TRACE") == "1":
            print(format_trace(result["trace"]))
    except ValueError as error:
        print(f"配置错误：{error}")


if __name__ == "__main__":
    run()
