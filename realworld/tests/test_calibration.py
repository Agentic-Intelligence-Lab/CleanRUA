import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from realworld.__main__ import main


class WorkspaceCalibrationTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "workstation.json"
        self.config = {"arms": {"left": {"driver": "lab:Arm", "tool": {"driver": "lab:Tool"}},
                                "right": {"driver": "lab:Other", "workspace_m": [[0, 0, 0], [1, 1, 1]]}},
                       "cameras": {"front": {"driver": "lab:Camera"}}}
        self.path.write_text(json.dumps(self.config))
        self.reader = Mock(spec=["state", "close"])
        self.reader.state.side_effect = [{"pose": [*xyz, 0, 0, 0]} for xyz in
                                        [(0.2, 0.3, -0.1), (-0.3, 0.5, 0.4), (0.1, -0.2, 0.7)]]
        load = patch("realworld.calibration.device", return_value=self.reader)
        self.device = load.start()
        self.addCleanup(load.stop)

    def cli(self, answers, *args):
        with patch("builtins.input", side_effect=answers):
            main(["calibrate-workspace", "--config", str(self.path), *args])

    def test_confirm_updates_only_selected_arm(self):
        self.cli(["", "", "", "done", "y"], "--arm", "left")
        expected = copy.deepcopy(self.config)
        expected["arms"]["left"]["workspace_m"] = [[-0.3, -0.2, -0.1], [0.2, 0.5, 0.7]]
        self.assertEqual(json.loads(self.path.read_text()), expected)
        self.device.assert_called_once_with(self.config["arms"]["left"])
        self.reader.close.assert_called_once()

    def test_single_arm_and_alternate_feedback_source(self):
        del self.config["arms"]["left"]
        self.config["arms"]["right"]["state_reader"] = {"driver": "lab:Feedback", "args": {"stream": 2}}
        self.path.write_text(json.dumps(self.config))
        self.cli(["", "", "done", "yes"])
        self.device.assert_called_once_with(self.config["arms"]["right"]["state_reader"])
        self.assertEqual(json.loads(self.path.read_text())["arms"]["right"]["workspace_m"],
                         [[-0.3, 0.3, -0.1], [0.2, 0.5, 0.4]])

    def test_cancel_preserves_original_bytes(self):
        original = self.path.read_bytes()
        for answers in (["q"], [KeyboardInterrupt()], [EOFError()], ["", "", "done", ""]):
            with self.subTest(answers=answers):
                self.cli(answers, "--arm", "left")
                self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(self.reader.close.call_count, 4)

    def test_invalid_and_flat_samples_do_not_form_a_box(self):
        self.reader.state.side_effect = [{"pose": [float("nan")] * 6}, {"pose": [0] * 6},
                                         {"pose": [0, 1, 1, 0, 0, 0]}]
        original = self.path.read_bytes()
        self.cli(["done", "", "", "", "done", "q"], "--arm", "left")
        self.assertEqual(self.path.read_bytes(), original)

    def test_dual_requires_side_and_never_accepts_motion(self):
        for args in ([], ["--arm", "left", "--enable-motion"]):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                self.cli([], *args)
        self.device.assert_not_called()

    def test_concurrent_edit_is_preserved(self):
        changed = json.dumps({**self.config, "revision": 2})
        answers = iter(["", "", "done", "y"])
        def answer(_prompt):
            cmd = next(answers)
            if cmd == "y":
                self.path.write_text(changed)
            return cmd
        with self.assertRaises(SystemExit):
            self.cli(answer, "--arm", "left")
        self.assertEqual(self.path.read_text(), changed)
