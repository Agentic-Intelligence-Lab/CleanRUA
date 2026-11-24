import base64
import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from realworld.model import configured_decider


class DecisionCLITest(unittest.TestCase):
    def test_chatgpt_only(self):
        with patch.dict(os.environ, {"CLEANRUA_DECISION_BACKEND":"codex"}), patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "", "Logged in using an API key\n")
            with self.assertRaisesRegex(RuntimeError, "ChatGPT account"):
                configured_decider([])
            self.assertEqual(run.call_count, 1)

    def test_images_schema_and_no_api_fallback(self):
        tools = [{"function":{"name":"finish"}}]
        def fake(args, **kw):
            self.assertNotIn("OPENAI_API_KEY", kw["env"])
            self.assertNotIn("CLEANRUA_API_KEY", kw["env"])
            if args[1] == "login":
                return subprocess.CompletedProcess(args, 0, "", "Logged in using ChatGPT\n")
            tmp = kw["cwd"]
            self.assertEqual([Path(args[i+1]).read_bytes() for i,a in enumerate(args) if a=="--image"],
                             [b"left",b"right"])
            self.assertEqual(args[args.index("--model")+1], "my-codex-model")
            self.assertIn('model_reasoning_effort="medium"', args)
            self.assertIn("forced_login_method=\"chatgpt\"", args)
            self.assertIn("[attached image 2]", kw["input"])
            self.assertNotIn("base64,", kw["input"])
            schema = json.loads((tmp/"action.schema.json").read_text())
            self.assertEqual(schema["properties"]["name"]["enum"], ["finish"])
            (tmp/"action.json").write_text('{"name":"finish","arguments":"{}"}')
            return subprocess.CompletedProcess(args,0,None,"")
        images = [{"type":"image_url","image_url":{"url":"data:image/png;base64,"+
                   base64.b64encode(data).decode()}} for data in (b"left", b"right")]
        with patch.dict(os.environ, {"CLEANRUA_DECISION_BACKEND":"codex","CLEANRUA_CODEX_MODEL":"my-codex-model","CLEANRUA_REASONING_EFFORT":"medium","OPENAI_API_KEY":"unused",
                                     "CLEANRUA_API_KEY":"unused"}), patch("subprocess.run", side_effect=fake):
            reply = configured_decider(tools)([{"role":"user","content":images}])
        self.assertEqual(reply["tool_calls"][0]["function"], {"name":"finish","arguments":"{}"})
