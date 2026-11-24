import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from realworld.__main__ import main


class RealMenuTest(unittest.TestCase):
    def test_model_request_and_report(self):
        env = {"CLEANRUA_API_KEY": "test",
               "CLEANRUA_MODEL_NAME": "configured-vlm"}
        cases = [(["--base-url", "https://provider.invalid/v1"], ["", "", "检查桌面", "", "", ""], "api", "configured-vlm", "default"),
                 (["direct", "--task", "检查桌面", "--backend", "local", "--model", "local-vlm",
                   "--reasoning-effort", "low"], [], "local", "local-vlm", "low")]
        for args, answers, backend, model, effort in cases:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp, "run")
                with patch.dict(os.environ, env, clear=True), patch("builtins.input", side_effect=answers), \
                        patch("realworld.real_decision.httpx.post") as post:
                    post.return_value.json.return_value = {"choices": [{"message": {"role": "assistant",
                        "tool_calls": [{"id": "1", "function": {"name": "finish", "arguments": "{}"}}]}}]}
                    main([*args, "--out", str(out)])
                request = post.call_args.kwargs["json"]
                self.assertEqual(request["model"], model)
                self.assertEqual(request.get("reasoning_effort", "default"), effort)
                self.assertEqual(request["messages"][1]["content"], "检查桌面")
                self.assertIn("image_url", request["messages"][-1]["content"][-1])
                if backend == "local":
                    self.assertEqual(post.call_args.kwargs["headers"], {})
                else:
                    self.assertNotIn("reasoning_effort", request)
                report = json.loads((out / "result.json").read_text())
                self.assertEqual([report[k] for k in ("decision_backend", "model", "reasoning_effort")],
                                 [backend, model, effort])
                self.assertTrue(report["agent_finished"])
                self.assertIsNone(report["task_success"])
