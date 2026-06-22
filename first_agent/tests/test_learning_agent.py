import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

from learning_agent import (
    build_profile,
    build_final_prompt,
    build_prompt,
    build_tool_instructions,
    create_openai_model,
    create_siliconflow_model,
    decide_next_step,
    decide_next_step_with_llm,
    execute_action,
    extract_preferences,
    format_trace,
    load_course_catalog,
    pick_model,
    parse_llm_response,
    parse_skills,
    read_chapter_excerpt,
    recommend_course_chapters,
    run,
    run_agent,
    run_agent_conversation,
    run_agent_loop,
    save_run_log,
    TOOL_SCHEMAS,
    validate_api_key,
    validate_decision,
)


class LearningAgentTest(unittest.TestCase):
    def test_frontend_dev_with_python_and_api_can_start_agent_project(self):
        result = decide_next_step(
            goal="agent",
            skills=["frontend", "python", "api"],
        )

        self.assertEqual(result["action"], "start_project")
        self.assertIn("做一个可展示的 agent 项目", result["message"])

    def test_frontend_dev_with_python_should_learn_api_next(self):
        result = decide_next_step(
            goal="agent",
            skills=["frontend", "python"],
        )

        self.assertEqual(result["action"], "learn_api")
        self.assertIn("补 API", result["message"])

    def test_frontend_dev_without_python_should_learn_python_first(self):
        result = decide_next_step(
            goal="agent",
            skills=["frontend"],
        )

        self.assertEqual(result["action"], "learn_python")
        self.assertIn("先补 Python", result["message"])

    def test_parse_skills_splits_comma_separated_input(self):
        result = parse_skills("frontend, python, API")

        self.assertEqual(result, ["frontend", "python", "API"])

    def test_parse_skills_ignores_empty_items(self):
        result = parse_skills("frontend, , python,")

        self.assertEqual(result, ["frontend", "python"])

    def test_build_profile_reads_goal_and_skills(self):
        with patch("builtins.input", side_effect=["agent", "frontend, python"]):
            profile = build_profile()

        self.assertEqual(profile["goal"], "agent")
        self.assertEqual(profile["skills"], ["frontend", "python"])

    def test_build_prompt_includes_goal_and_skills(self):
        prompt = build_prompt(
            goal="agent",
            skills=["frontend", "python"],
        )

        self.assertIn("agent", prompt)
        self.assertIn("frontend", prompt)
        self.assertIn("python", prompt)
        self.assertIn("learn_python", prompt)
        self.assertIn("learn_api", prompt)
        self.assertIn("start_project", prompt)
        self.assertIn("plan_learning", prompt)
        self.assertIn("clarify_goal", prompt)
        self.assertIn("args", prompt)

    def test_build_prompt_includes_memory_clarifications(self):
        prompt = build_prompt(
            goal="agent",
            skills=["frontend", "python"],
            memory={
                "clarifications": [
                    {"question": "项目方向？", "answer": "我想做前端 agent"}
                ]
            },
        )

        self.assertIn("澄清记录", prompt)
        self.assertIn("我想做前端 agent", prompt)

    def test_build_tool_instructions_uses_tool_schemas(self):
        instructions = build_tool_instructions(TOOL_SCHEMAS)

        self.assertIn("learn_api", instructions)
        self.assertIn("用户会 Python", instructions)
        self.assertIn("clarify_goal", instructions)

    def test_parse_llm_response_reads_json_action_and_message(self):
        result = parse_llm_response(
            '{"action": "learn_api", "message": "补 API，再做 agent"}'
        )

        self.assertEqual(result["action"], "learn_api")
        self.assertEqual(result["message"], "补 API，再做 agent")
        self.assertEqual(result["args"], {})

    def test_parse_llm_response_keeps_args(self):
        result = parse_llm_response(
            '{"action": "learn_api", "message": "补 API", "args": {"focus": "backend"}}'
        )

        self.assertEqual(result["args"]["focus"], "backend")

    def test_validate_decision_rejects_unknown_action(self):
        with self.assertRaisesRegex(ValueError, "未知 action"):
            validate_decision({
                "action": "invent_tool",
                "message": "随便调用",
                "args": {},
            })

    def test_validate_decision_rejects_non_object_args(self):
        with self.assertRaisesRegex(ValueError, "args 必须是对象"):
            validate_decision({
                "action": "learn_api",
                "message": "补 API",
                "args": "backend",
            })

    def test_validate_decision_adds_default_args(self):
        result = validate_decision({
            "action": "learn_api",
            "message": "补 API",
        })

        self.assertEqual(result["args"], {})

    def test_parse_llm_response_rejects_non_json_text(self):
        with self.assertRaisesRegex(ValueError, "模型没有返回合法 JSON"):
            parse_llm_response("可以先补 API")

    def test_parse_llm_response_accepts_json_markdown_block(self):
        result = parse_llm_response(
            '```json\n{"action": "learn_python", "message": "先学习 Python"}\n```'
        )

        self.assertEqual(result["action"], "learn_python")

    def test_parse_llm_response_extracts_json_from_extra_text(self):
        result = parse_llm_response(
            '好的，建议如下：{"action": "learn_python", "message": "先学习 Python"}'
        )

        self.assertEqual(result["message"], "先学习 Python")

    def test_decide_next_step_with_llm_calls_model_with_prompt(self):
        prompts = []

        def fake_model(prompt):
            prompts.append(prompt)
            return '{"action": "learn_api", "message": "补 API，再做 agent"}'

        result = decide_next_step_with_llm(
            goal="agent",
            skills=["frontend", "python"],
            model=fake_model,
        )

        self.assertEqual(result["action"], "learn_api")
        self.assertEqual(result["message"], "补 API，再做 agent")
        self.assertEqual(len(prompts), 1)
        self.assertIn("frontend", prompts[0])

    def test_execute_action_runs_known_tool(self):
        result = execute_action({
            "action": "learn_api",
            "message": "补 API",
        })

        self.assertIn("第六章", result["tool_result"])

    def test_execute_action_reports_unknown_tool(self):
        result = execute_action({
            "action": "unknown",
            "message": "未知动作",
        })

        self.assertIn("还没有工具", result["tool_result"])

    def test_execute_action_handles_clarify_goal(self):
        result = execute_action({
            "action": "clarify_goal",
            "message": "需要明确目标",
        })

        self.assertIn("可以继续说清楚", result["tool_result"])

    def test_execute_action_builds_learning_plan(self):
        result = execute_action({
            "action": "plan_learning",
            "message": "先按学习路线系统补齐",
            "skills": ["frontend", "python"],
            "memory": {"preferences": {"goal_type": "study", "focus": "frontend"}},
        })

        self.assertIn("3 步学习计划", result["tool_result"])
        self.assertIn("每天 1 小时", result["tool_result"])
        self.assertIn("第六章", result["tool_result"])

    def test_run_agent_combines_model_decision_and_tool_result(self):
        def fake_model(_prompt):
            return '{"action": "learn_api", "message": "补 API"}'

        result = run_agent(
            goal="agent",
            skills=["frontend", "python"],
            model=fake_model,
        )

        self.assertEqual(result["action"], "learn_api")
        self.assertEqual(result["message"], "补 API")
        self.assertIn("第六章", result["tool_result"])

    def test_run_agent_passes_args_to_tool(self):
        def fake_model(_prompt):
            return '{"action": "learn_api", "message": "补 API", "args": {"focus": "backend"}}'

        result = run_agent(
            goal="agent",
            skills=["frontend", "python"],
            model=fake_model,
        )

        self.assertIn("后端/API", result["tool_result"])

    def test_build_final_prompt_includes_tool_result(self):
        prompt = build_final_prompt(
            goal="agent",
            skills=["frontend", "python"],
            observation={
                "message": "补 API",
                "tool_result": "推荐第六章",
            },
        )

        self.assertIn("补 API", prompt)
        self.assertIn("推荐第六章", prompt)
        self.assertIn("最终回答", prompt)

    def test_build_final_prompt_includes_memory_preferences(self):
        prompt = build_final_prompt(
            goal="agent",
            skills=["frontend", "python"],
            observation={
                "message": "补 API",
                "tool_result": "推荐第六章",
                "memory": {"preferences": {"goal_type": "portfolio", "focus": "frontend"}},
            },
        )

        self.assertIn("作品集", prompt)
        self.assertIn("前端", prompt)

    def test_run_agent_loop_uses_second_model_call_for_final_answer(self):
        prompts = []

        def fake_model(prompt):
            prompts.append(prompt)
            if len(prompts) == 1:
                return '{"action": "learn_api", "message": "补 API"}'
            return "最终建议：先读第六章，再做一个 API 小练习。"

        result = run_agent_loop(
            goal="agent",
            skills=["frontend", "python"],
            model=fake_model,
        )

        self.assertEqual(len(prompts), 2)
        self.assertIn("final_answer", result)
        self.assertIn("第六章", result["final_answer"])
        self.assertIn("tool_result", result)

    def test_run_agent_loop_returns_trace_steps(self):
        def fake_model(prompt):
            if "最终回答" in prompt:
                return "最终建议：先读第六章。"
            return '{"action": "learn_api", "message": "补 API"}'

        result = run_agent_loop(
            goal="agent",
            skills=["frontend", "python"],
            model=fake_model,
        )

        self.assertEqual(
            [step["step"] for step in result["trace"]],
            ["decision", "tool", "final"],
        )
        self.assertEqual(result["trace"][0]["action"], "learn_api")
        self.assertIn("第六章", result["trace"][1]["tool_result"])
        self.assertIn("最终建议", result["trace"][2]["final_answer"])

    def test_run_agent_conversation_retries_after_clarification(self):
        prompts = []

        def fake_model(prompt):
            prompts.append(prompt)
            if len(prompts) == 1:
                return '{"action": "clarify_goal", "message": "请说明项目方向"}'
            if "最终回答" in prompt:
                return "最终建议：先读第六章。"
            return '{"action": "learn_api", "message": "补 API"}'

        result = run_agent_conversation(
            goal="agent",
            skills=["frontend", "python"],
            model=fake_model,
            ask_user=lambda question: "我想做求职作品集里的前端 agent",
        )

        self.assertEqual(result["action"], "learn_api")
        self.assertIn("第六章", result["tool_result"])
        self.assertIn("我想做求职作品集", prompts[1])
        self.assertEqual(result["trace"][0]["step"], "clarification")
        self.assertEqual(
            result["memory"]["clarifications"][0]["answer"],
            "我想做求职作品集里的前端 agent",
        )
        self.assertEqual(result["memory"]["preferences"]["goal_type"], "portfolio")
        self.assertEqual(result["memory"]["preferences"]["focus"], "frontend")

    def test_run_agent_uses_memory_in_second_round(self):
        prompts = []

        def fake_model(prompt):
            prompts.append(prompt)
            if len(prompts) == 1:
                return '{"action": "clarify_goal", "message": "请说明项目方向"}'
            if "最终回答" in prompt:
                return "最终建议：先读第六章。"
            return '{"action": "learn_api", "message": "补 API"}'

        run_agent_conversation(
            goal="agent",
            skills=["frontend", "python"],
            model=fake_model,
            ask_user=lambda question: "我想做前端 agent",
        )

        self.assertIn("澄清记录", prompts[1])
        self.assertIn("我想做前端 agent", prompts[1])

    def test_format_trace_outputs_debug_lines(self):
        trace_text = format_trace([
            {"step": "decision", "action": "learn_api", "message": "补 API", "args": {}},
            {"step": "tool", "action": "learn_api", "tool_result": "推荐第六章"},
            {"step": "final", "final_answer": "最终建议"},
        ])

        self.assertIn("[decision]", trace_text)
        self.assertIn("learn_api", trace_text)
        self.assertIn("[tool]", trace_text)
        self.assertIn("[final]", trace_text)

    def test_run_prints_trace_when_show_trace_is_enabled(self):
        with patch("builtins.input", side_effect=["agent", "frontend, python"]):
            with patch.dict("os.environ", {"SHOW_TRACE": "1"}, clear=True):
                with patch("builtins.print") as print_mock:
                    run()

        printed_text = "\n".join(call.args[0] for call in print_mock.call_args_list)
        self.assertIn("[decision]", printed_text)
        self.assertIn("[tool]", printed_text)

    def test_save_run_log_writes_json_file(self):
        with TemporaryDirectory() as tmpdir:
            path = save_run_log(
                profile={"goal": "agent", "skills": ["frontend", "python"]},
                result={
                    "final_answer": "最终建议",
                    "trace": [{"step": "decision", "action": "learn_api"}],
                    "memory": {"clarifications": [{"answer": "作品集"}]},
                },
                runs_dir=Path(tmpdir),
            )

            self.assertTrue(path.exists())
            self.assertEqual(path.suffix, ".json")
            content = path.read_text(encoding="utf-8")
            self.assertIn("最终建议", content)
            self.assertIn("learn_api", content)
            self.assertIn("作品集", content)

    def test_load_course_catalog_reads_sidebar_chapters(self):
        chapters = load_course_catalog()

        self.assertIn("第一章 初识智能体", chapters)
        self.assertIn("第七章 构建你的Agent框架", chapters)

    def test_read_chapter_excerpt_reads_local_markdown(self):
        excerpt = read_chapter_excerpt("第六章 框架开发实践", max_lines=4)

        self.assertIn("第六章 框架开发实践", excerpt)
        self.assertIn("智能体框架", excerpt)

    def test_recommend_course_chapters_for_learn_api(self):
        result = recommend_course_chapters({"action": "learn_api"})

        self.assertIn("第六章", result)
        self.assertIn("第七章", result)
        self.assertIn("内容摘录", result)

    def test_recommend_course_chapters_uses_focus_arg(self):
        result = recommend_course_chapters({
            "action": "learn_api",
            "args": {"focus": "backend"},
        })

        self.assertIn("后端/API", result)

    def test_recommend_course_chapters_uses_memory_focus(self):
        result = recommend_course_chapters({
            "action": "learn_api",
            "args": {},
            "memory": {"preferences": {"focus": "backend"}},
        })

        self.assertIn("后端/API", result)

    def test_extract_preferences_reads_goal_type_and_focus(self):
        preferences = extract_preferences("我想做求职作品集里的前端 agent")

        self.assertEqual(preferences["goal_type"], "portfolio")
        self.assertEqual(preferences["focus"], "frontend")

    def test_recommend_course_chapters_for_start_project_starts_with_framework_chapters(self):
        result = recommend_course_chapters({"action": "start_project"})

        self.assertIn("第六章", result)
        self.assertIn("第七章", result)
        self.assertIn("第十三章", result)
        self.assertLess(result.index("第六章"), result.index("第十三章"))

    def test_recommend_course_chapters_for_plan_learning_returns_three_step_plan(self):
        result = recommend_course_chapters({
            "action": "plan_learning",
            "skills": ["frontend", "python"],
            "memory": {"preferences": {"goal_type": "study", "focus": "frontend"}},
        })

        self.assertIn("3 步学习计划", result)
        self.assertIn("每天 1 小时", result)
        self.assertIn("前端 Agent", result)
        self.assertIn("第六章", result)

    def test_create_openai_model_calls_responses_api(self):
        create_mock = MagicMock(
            return_value=SimpleNamespace(
                output_text='{"action": "learn_api", "message": "补 API"}'
            )
        )

        class FakeOpenAI:
            def __init__(self):
                self.responses = SimpleNamespace(create=create_mock)

        with patch.dict("sys.modules", {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}):
            model = create_openai_model(model_name="gpt-test")
            result = model("hello")

        self.assertEqual(result, '{"action": "learn_api", "message": "补 API"}')
        create_mock.assert_called_once_with(model="gpt-test", input="hello")

    def test_create_siliconflow_model_calls_chat_completions_api(self):
        create_mock = MagicMock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"action": "learn_api", "message": "补 API"}'
                        )
                    )
                ]
            )
        )
        clients = []

        class FakeOpenAI:
            def __init__(self, api_key=None, base_url=None):
                self.api_key = api_key
                self.base_url = base_url
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=create_mock)
                )
                clients.append(self)

        with patch.dict("sys.modules", {"openai": SimpleNamespace(OpenAI=FakeOpenAI)}):
            model = create_siliconflow_model(
                api_key="test-key",
                model_name="siliconflow-test",
            )
            result = model("hello")

        self.assertEqual(result, '{"action": "learn_api", "message": "补 API"}')
        self.assertEqual(clients[0].api_key, "test-key")
        self.assertEqual(clients[0].base_url, "https://api.siliconflow.cn/v1")
        create_mock.assert_called_once_with(
            model="siliconflow-test",
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
        )

    def test_pick_model_prefers_siliconflow_key_over_openai_key(self):
        with patch("learning_agent.create_siliconflow_model") as create_siliconflow:
            create_siliconflow.return_value = lambda _prompt: (
                '{"action": "learn_api", "message": "补 API"}'
            )

            model = pick_model(
                goal="agent",
                skills=["frontend", "python"],
                siliconflow_api_key="siliconflow-key",
                openai_api_key="openai-key",
            )

        result = parse_llm_response(model("ignored prompt"))
        self.assertEqual(result["action"], "learn_api")
        create_siliconflow.assert_called_once_with(api_key="siliconflow-key")

    def test_pick_model_uses_rule_based_fallback_without_api_key(self):
        model = pick_model(
            goal="agent",
            skills=["frontend", "python"],
            siliconflow_api_key=None,
            openai_api_key=None,
        )

        result = parse_llm_response(model("ignored prompt"))
        self.assertEqual(result["action"], "learn_api")

    def test_rule_based_model_returns_text_for_final_prompt(self):
        model = pick_model(
            goal="agent",
            skills=["frontend", "python"],
            siliconflow_api_key=None,
            openai_api_key=None,
        )

        result = model("请基于工具执行结果，给用户一个简短、自然、可执行的最终回答。")

        self.assertIn("最终建议", result)

    def test_rule_based_model_returns_portfolio_flavored_final_text(self):
        model = pick_model(
            goal="agent",
            skills=["frontend", "python"],
            siliconflow_api_key=None,
            openai_api_key=None,
        )

        result = model("请基于工具执行结果，给用户一个简短、自然、可执行的最终回答。用户偏好：作品集 前端")

        self.assertIn("作品集", result)

    def test_rule_based_model_returns_learning_plan_action_for_study_intent(self):
        model = pick_model(
            goal="agent",
            skills=["frontend", "python"],
            siliconflow_api_key=None,
            openai_api_key=None,
        )

        result = parse_llm_response(
            model(
                "你是一个 agent 学习顾问。\n"
                "用户目标：agent\n"
                "用户技能：frontend, python\n"
                "澄清记录：\n"
                "- 问题：方向？\n"
                "  回答：我想先系统学习 agent 原理，再做前端 Agent\n"
            )
        )

        self.assertEqual(result["action"], "plan_learning")

    def test_validate_api_key_rejects_non_ascii_placeholder(self):
        with self.assertRaisesRegex(ValueError, "SILICONFLOW_API_KEY"):
            validate_api_key("SILICONFLOW_API_KEY", "你的 key")

    def test_run_prints_friendly_error_for_invalid_api_key(self):
        with patch("builtins.input", side_effect=["agent", "py"]):
            with patch.dict("os.environ", {"SILICONFLOW_API_KEY": "你的 key"}):
                with patch("builtins.print") as print_mock:
                    run()

        print_mock.assert_called_once()
        self.assertIn("SILICONFLOW_API_KEY", print_mock.call_args.args[0])

    def test_run_prints_friendly_error_for_invalid_model_response(self):
        with patch("builtins.input", side_effect=["agent", "python"]):
            with patch("learning_agent.pick_model", return_value=lambda _prompt: ""):
                with patch("builtins.print") as print_mock:
                    run()

        print_mock.assert_called_once()
        self.assertIn("模型没有返回合法 JSON", print_mock.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
