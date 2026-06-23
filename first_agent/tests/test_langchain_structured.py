import unittest
from unittest.mock import MagicMock, patch

import langchain_structured


class LangchainStructuredTest(unittest.TestCase):
    def test_build_llm_requires_siliconflow_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "SILICONFLOW_API_KEY"):
                langchain_structured.build_llm()

    def test_build_prompt_requests_json_fields(self):
        prompt = langchain_structured.build_prompt("React hooks 入门")

        self.assertIn("JSON", prompt)
        self.assertIn("topic", prompt)
        self.assertIn("difficulty", prompt)
        self.assertIn("React hooks 入门", prompt)

    def test_parse_task_json_validates_model_output(self):
        task = langchain_structured.parse_task_json(
            '{"topic":"React Hooks","difficulty":"easy","action":"写三个 useState 小练习","keywords":["useState","useEffect"]}'
        )

        self.assertEqual(task.topic, "React Hooks")
        self.assertEqual(task.difficulty, "easy")
        self.assertIn("useState", task.keywords)

    def test_run_demo_uses_llm_and_parses_json(self):
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = (
            '{"topic":"Vite","difficulty":"easy","action":"搭一个最小项目","keywords":["vite","dev server"]}'
        )

        with patch("langchain_structured.build_llm", return_value=fake_llm):
            task = langchain_structured.run_demo("Vite 入门")

        fake_llm.invoke.assert_called_once()
        self.assertEqual(task.topic, "Vite")
        self.assertEqual(task.difficulty, "easy")


if __name__ == "__main__":
    unittest.main()
