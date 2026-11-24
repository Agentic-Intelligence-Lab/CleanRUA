import os
import unittest
from unittest.mock import patch

from realworld.model import mllm


class ApiCredentialsTest(unittest.TestCase):
    def test_provider_key_is_selected_by_destination(self):
        env = {"OPENAI_API_KEY": "openai-test", "DASHSCOPE_API_KEY": "dashscope-test",
               "OPENROUTER_API_KEY": "openrouter-test", "CLEANRUA_MODEL_NAME": "vlm"}
        endpoints = [("https://api.openai.com/v1", "openai-test"),
                     ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "dashscope-test"),
                     ("https://openrouter.ai/api/v1", "openrouter-test")]
        for url, key in endpoints:
            with self.subTest(url=url), patch.dict(os.environ, {**env, "CLEANRUA_BASE_URL": url}, clear=True), \
                    patch("realworld.model.httpx.post") as post:
                post.return_value.json.return_value = {"choices": [{"message": {"tool_calls": []}}]}
                mllm([], [])
                self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer " + key)
                os.environ["CLEANRUA_API_KEY"] = "explicit-test"
                mllm([], [])
                self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer explicit-test")

    def test_unknown_origin_never_receives_provider_key(self):
        for url in ("https://proxy.invalid/v1", "http://api.openai.com/v1", "https://api.openai.com:8443/v1"):
            with self.subTest(url=url), patch.dict(os.environ, {"CLEANRUA_BASE_URL": url,
                    "CLEANRUA_MODEL_NAME": "vlm", "OPENAI_API_KEY": "test"}, clear=True), \
                    patch("realworld.model.httpx.post") as post:
                with self.assertRaisesRegex(ValueError, "CLEANRUA_API_KEY"):
                    mllm([], [])
                post.assert_not_called()
