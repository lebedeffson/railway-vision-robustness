import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from download_osdar23_direct import sequence_is_complete
from stream_extract_osdar23 import stream_extract


class StreamExtractionTest(unittest.TestCase):
    def test_extracts_referenced_frames_from_file_stream(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            raw_dir = workspace / "raw"
            archive_path = workspace / "sample.zip"
            sequence = "sample_1.1"
            frames = {
                str(index): {
                    "frame_properties": {
                        "streams": {
                            "rgb_highres_center": {
                                "uri": f"/rgb_highres_center/{index:03d}.png"
                            }
                        }
                    }
                }
                for index in range(2)
            }
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(
                    f"{sequence}_labels.json",
                    json.dumps({"openlabel": {"frames": frames}}),
                )
                archive.writestr("rgb_highres_center/000.png", b"first")
                archive.writestr("rgb_highres_center/001.png", b"second")

            curl_status, archive_status, missing = stream_extract(
                sequence,
                archive_path.as_uri(),
                raw_dir=raw_dir,
                download_ip=None,
                rate_limit=None,
            )

            self.assertEqual(curl_status, 0)
            self.assertEqual(archive_status, 0)
            self.assertEqual(missing, [])
            self.assertTrue(sequence_is_complete(sequence, raw_dir=raw_dir))

    def test_does_not_overwrite_an_existing_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            raw_dir = workspace / "raw"
            archive_path = workspace / "sample.zip"
            sequence = "sample_1.1"
            root = raw_dir / sequence
            camera = root / "rgb_highres_center"
            camera.mkdir(parents=True)
            frames = {
                str(index): {
                    "frame_properties": {
                        "streams": {
                            "rgb_highres_center": {
                                "uri": f"/rgb_highres_center/{index:03d}.png"
                            }
                        }
                    }
                }
                for index in range(2)
            }
            labels = json.dumps({"openlabel": {"frames": frames}})
            (root / f"{sequence}_labels.json").write_text(labels, encoding="utf-8")
            (camera / "000.png").write_bytes(b"keep")
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(f"{sequence}_labels.json", labels)
                archive.writestr("rgb_highres_center/000.png", b"replace")
                archive.writestr("rgb_highres_center/001.png", b"recover")

            _, _, missing = stream_extract(
                sequence,
                archive_path.as_uri(),
                raw_dir=raw_dir,
                download_ip=None,
                rate_limit=None,
            )

            self.assertEqual(missing, [])
            self.assertEqual((camera / "000.png").read_bytes(), b"keep")
            self.assertEqual((camera / "001.png").read_bytes(), b"recover")


if __name__ == "__main__":
    unittest.main()
