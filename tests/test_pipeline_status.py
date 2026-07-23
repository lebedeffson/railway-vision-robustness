import tempfile
import unittest
from pathlib import Path

from pipeline_status import STAGES, load, update


class PipelineStatusTest(unittest.TestCase):
    def test_stage_contract_and_resume_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = root / "pipeline_status.json"
            input_file = root / "input.txt"
            output_file = root / "output.txt"
            input_file.write_text("input", encoding="utf-8")
            output_file.write_text("output", encoding="utf-8")
            update(
                "attack_generation", "running",
                inputs=[input_file], path=status,
            )
            update(
                "attack_generation", "success",
                inputs=[input_file], outputs=[output_file], path=status,
            )
            payload = load(status)
            self.assertEqual(set(payload["stages"]), set(STAGES))
            stage = payload["stages"]["attack_generation"]
            self.assertEqual(stage["status"], "success")
            self.assertIsNotNone(stage["started_at"])
            self.assertIsNotNone(stage["finished_at"])
            self.assertEqual(len(stage["input_hash"]), 64)
            self.assertEqual(stage["output_files"], [str(output_file)])
            self.assertIsNone(stage["error"])


if __name__ == "__main__":
    unittest.main()
