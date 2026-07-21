import json
import tempfile
import unittest
from pathlib import Path

from download_osdar23_direct import (
    expected_camera_paths,
    load_frame_exclusions,
    missing_sequence_files,
    sequence_is_complete,
    sequence_is_usable,
)


class DownloadCompletenessTest(unittest.TestCase):
    def make_sequence(self, raw_dir: Path, uris: list[str]) -> tuple[str, Path]:
        sequence = "sample_1.1"
        root = raw_dir / sequence
        root.mkdir(parents=True)
        frames = {
            str(index): {
                "frame_properties": {
                    "streams": {"rgb_highres_center": {"uri": uri}}
                }
            }
            for index, uri in enumerate(uris)
        }
        labels = {"openlabel": {"frames": frames}}
        (root / f"{sequence}_labels.json").write_text(
            json.dumps(labels), encoding="utf-8"
        )
        return sequence, root

    def test_all_referenced_frames_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            sequence, root = self.make_sequence(
                raw_dir,
                [
                    "/rgb_highres_center/000.png",
                    "/rgb_highres_center/001.png",
                ],
            )
            camera = root / "rgb_highres_center"
            camera.mkdir()
            (camera / "000.png").touch()

            self.assertFalse(sequence_is_complete(sequence, raw_dir=raw_dir))
            self.assertEqual(missing_sequence_files(sequence, raw_dir), [camera / "001.png"])

            (camera / "001.png").touch()
            self.assertTrue(sequence_is_complete(sequence, raw_dir=raw_dir))
            self.assertEqual(len(expected_camera_paths(sequence, raw_dir)), 2)

    def test_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            sequence, _ = self.make_sequence(raw_dir, ["../outside.png"])
            self.assertFalse(sequence_is_complete(sequence, raw_dir=raw_dir))

    def test_explicit_missing_frame_can_be_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            sequence, root = self.make_sequence(
                raw_dir, ["/rgb_highres_center/000.png"]
            )
            policy = raw_dir / "policy.json"
            policy.write_text(
                json.dumps({
                    "excluded_relative_paths": [
                        f"{sequence}/rgb_highres_center/000.png"
                    ]
                }),
                encoding="utf-8",
            )
            self.assertFalse(sequence_is_complete(sequence, raw_dir=raw_dir))
            self.assertTrue(sequence_is_usable(sequence, raw_dir, policy))
            self.assertEqual(
                load_frame_exclusions(policy),
                {f"{sequence}/rgb_highres_center/000.png"},
            )
            self.assertTrue(root.is_dir())


if __name__ == "__main__":
    unittest.main()
