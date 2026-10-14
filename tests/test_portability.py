import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from grpo_agent.config import load_config, model_source


class PortabilityTests(unittest.TestCase):
    def test_model_environment_overrides_yaml(self):
        cfg = {"name": "repo/default", "local_path": None, "allow_download": False}
        env = {
            "MODEL_PATH": "/cloud/models/qwen",
            "TOKENIZER_PATH": "/cloud/tokenizer",
            "ALLOW_MODEL_DOWNLOAD": "true",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertEqual(
                model_source(cfg),
                ("/cloud/models/qwen", "/cloud/tokenizer", True),
            )

    def test_output_root_redirects_mutable_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"GRPO_OUTPUT_ROOT": directory}, clear=False):
                cfg = load_config("configs/baseline.yaml")
            root = Path(directory)
            self.assertEqual(Path(cfg["paths"]["trace_dir"]), root / "logs" / "traces")
            self.assertEqual(Path(cfg["paths"]["evaluation_dir"]), root / "logs" / "evaluation")
            self.assertTrue(Path(cfg["training"]["output_dir"]).is_relative_to(root))


if __name__ == "__main__":
    unittest.main()
