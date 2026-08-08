import json
import errno
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import ray_batch
from normalcrafter.utils import convert_normalcrafter_to_udn, decode_udn_rgb


class BatchStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input_root = self.root / "inputs"
        self.output_root = self.root / "outputs"
        self.input_root.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def make_video_placeholder(
        self, relative_path: str, content: bytes = b"video"
    ) -> Path:
        path = self.input_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def config(
        self,
        save_npz: bool = False,
        normal_standard: str = "normalcrafter",
        flip_y: bool = False,
    ):
        return {
            "model": {
                "unet_path": "model-a",
                "pre_train_path": "model-b",
                "cpu_offload": "model",
            },
            "inference": {
                "window_size": 14,
                "time_step_size": 10,
                "process_length": 14,
                "decode_chunk_size": 7,
                "max_res": 512,
                "dataset": "open",
                "target_fps": -1,
                "seed": 42,
                "save_npz": save_npz,
                "normal_standard": normal_standard,
                "flip_y": flip_y,
            },
            "batch": {"output_mode": "full"},
        }

    def test_discovery_is_recursive_deduplicated_and_excludes_output(self):
        first = self.make_video_placeholder("a/clip.mp4")
        second = self.make_video_placeholder("b/movie.MOV")
        self.make_video_placeholder("ignored.txt")
        nested_output = self.input_root / "generated"
        nested_output.mkdir()
        (nested_output / "result.mp4").write_bytes(b"output")

        found = ray_batch.discover_inputs(
            [str(self.input_root), str(first)], [], nested_output
        )

        self.assertEqual(
            [item.path for item in found], [first.resolve(), second.resolve()]
        )
        self.assertEqual(
            [item.relative_path.as_posix() for item in found],
            ["a/clip.mp4", "b/movie.MOV"],
        )

    def test_input_list_paths_are_relative_to_list_file(self):
        video = self.make_video_placeholder("nested/clip.mp4")
        input_list = self.input_root / "videos.txt"
        input_list.write_text("# comment\nnested/clip.mp4\n", encoding="utf-8")

        found = ray_batch.discover_inputs([], [str(input_list)], self.output_root)

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].path, video.resolve())

    def test_job_id_changes_with_input_or_inference_identity(self):
        video = self.make_video_placeholder("clip.mp4")
        config = self.config()
        first = ray_batch.compute_job_id(video, config)
        self.assertEqual(first, ray_batch.compute_job_id(video, config))

        changed_config = self.config()
        changed_config["inference"]["seed"] = 7
        self.assertNotEqual(first, ray_batch.compute_job_id(video, changed_config))

        flipped_config = self.config(flip_y=True)
        self.assertNotEqual(first, ray_batch.compute_job_id(video, flipped_config))

        video.write_bytes(b"different video bytes")
        os.utime(video, None)
        self.assertNotEqual(first, ray_batch.compute_job_id(video, config))

    def test_jobs_preserve_relative_path_and_add_hash(self):
        video = self.make_video_placeholder("nested/clip.mp4")
        inputs = [ray_batch.InputSpec(video.resolve(), Path("nested/clip.mp4"))]

        job = ray_batch.build_jobs(inputs, self.config())[0]

        self.assertTrue(job.output_files[0][1].startswith("nested/clip_"))
        self.assertTrue(job.output_files[0][1].endswith("_input.mp4"))
        self.assertTrue(job.output_files[1][1].endswith("_vis.mp4"))

    def test_udn_job_declares_canonical_artifacts(self):
        video = self.make_video_placeholder("nested/clip.mp4")
        inputs = [ray_batch.InputSpec(video.resolve(), Path("nested/clip.mp4"))]

        job = ray_batch.build_jobs(
            inputs, self.config(save_npz=True, normal_standard="udn-v2")
        )[0]

        self.assertEqual(
            [kind for kind, _ in job.output_files],
            ["video", "video", "udn_npz", "udn_manifest"],
        )
        self.assertTrue(job.output_files[1][1].endswith("_normal_udn_vis.mp4"))
        self.assertTrue(job.output_files[2][1].endswith("_normal_udn.npz"))
        self.assertTrue(job.output_files[3][1].endswith("_manifest.json"))

    def test_video_only_job_uses_source_relative_mp4_name(self):
        video = self.make_video_placeholder("nested/clip.mp4")
        inputs = [ray_batch.InputSpec(video.resolve(), Path("nested/clip.mp4"))]
        config = self.config(normal_standard="udn-v2")
        config["batch"]["output_mode"] = "normal-video-only"

        job = ray_batch.build_jobs(inputs, config)[0]

        self.assertEqual(job.output_files, (("video", "nested/clip.mp4"),))

    def test_flip_y_cli_defaults_false_and_can_be_enabled(self):
        parser = ray_batch.build_parser()
        base = ["--input", "clip.mp4", "--output", "outputs"]

        default = parser.parse_args(base)
        flipped = parser.parse_args([*base, "--flip-y"])
        explicit_default = parser.parse_args([*base, "--no-flip-y"])

        self.assertFalse(default.flip_y)
        self.assertTrue(flipped.flip_y)
        self.assertFalse(explicit_default.flip_y)

    def test_atomic_json_and_valid_done_marker_drive_resume(self):
        video = self.make_video_placeholder("clip.mp4")
        spec = ray_batch.InputSpec(video.resolve(), Path("clip.mp4"))
        job = ray_batch.build_jobs([spec], self.config())[0]
        for _, relative_path in job.output_files:
            output = self.output_root / relative_path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"valid")
        marker = {
            "schema_version": ray_batch.SCHEMA_VERSION,
            "job_id": job.job_id,
            "outputs": [path for _, path in job.output_files],
            "output_sizes": {
                relative_path: (self.output_root / relative_path).stat().st_size
                for _, relative_path in job.output_files
            },
            "output_mtime_ns": {
                relative_path: (self.output_root / relative_path).stat().st_mtime_ns
                for _, relative_path in job.output_files
            },
        }
        marker_path = ray_batch._state_path(self.output_root, "done", job.job_id)
        ray_batch.atomic_write_json(marker_path, marker)

        with mock.patch("ray_batch.validate_artifact", return_value=(True, "ok")):
            pending, completed, failed = ray_batch.classify_jobs(
                [job], self.output_root, self.output_root, retry_failed=False
            )

        self.assertEqual(pending, [])
        self.assertEqual(completed, [job])
        self.assertEqual(failed, [])
        self.assertEqual(json.loads(marker_path.read_text())["job_id"], job.job_id)
        self.assertEqual(list(marker_path.parent.glob("*.tmp-*")), [])

    def test_finalize_artifact_falls_back_across_filesystems(self):
        source = self.root / "partial" / "normal.mp4"
        destination = self.root / "outputs" / "normal.mp4"
        source.parent.mkdir()
        source.write_bytes(b"predicted-normal")
        real_replace = os.replace

        def cross_device_once(raw_source, raw_destination):
            if Path(raw_source) == source:
                raise OSError(errno.EXDEV, "Invalid cross-device link")
            return real_replace(raw_source, raw_destination)

        with mock.patch("ray_batch.os.replace", side_effect=cross_device_once):
            ray_batch.finalize_artifact(source, destination)

        self.assertFalse(source.exists())
        self.assertEqual(destination.read_bytes(), b"predicted-normal")
        self.assertEqual(list(destination.parent.glob(".*.tmp-*")), [])

    def test_replaced_same_size_output_is_requeued(self):
        video = self.make_video_placeholder("clip.mp4")
        spec = ray_batch.InputSpec(video.resolve(), Path("clip.mp4"))
        job = ray_batch.build_jobs([spec], self.config())[0]
        for _, relative_path in job.output_files:
            output = self.output_root / relative_path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"valid")
        marker = {
            "schema_version": ray_batch.SCHEMA_VERSION,
            "job_id": job.job_id,
            "outputs": [path for _, path in job.output_files],
            "output_sizes": {
                path: (self.output_root / path).stat().st_size
                for _, path in job.output_files
            },
            "output_mtime_ns": {
                path: (self.output_root / path).stat().st_mtime_ns
                for _, path in job.output_files
            },
        }
        marker_path = ray_batch._state_path(self.output_root, "done", job.job_id)
        ray_batch.atomic_write_json(marker_path, marker)
        changed = self.output_root / job.output_files[0][1]
        stat = changed.stat()
        os.utime(changed, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))

        with mock.patch("ray_batch.validate_artifact", return_value=(True, "ok")):
            pending, completed, failed = ray_batch.classify_jobs(
                [job], self.output_root, self.output_root, retry_failed=False
            )

        self.assertEqual(pending, [job])
        self.assertEqual(completed, [])
        self.assertEqual(failed, [])

    def test_invalid_done_marker_is_removed_and_requeued(self):
        video = self.make_video_placeholder("clip.mp4")
        spec = ray_batch.InputSpec(video.resolve(), Path("clip.mp4"))
        job = ray_batch.build_jobs([spec], self.config())[0]
        marker_path = ray_batch._state_path(self.output_root, "done", job.job_id)
        ray_batch.atomic_write_json(marker_path, {"job_id": "wrong"})

        pending, completed, failed = ray_batch.classify_jobs(
            [job], self.output_root, self.output_root, retry_failed=False
        )

        self.assertEqual(pending, [job])
        self.assertEqual(completed, [])
        self.assertEqual(failed, [])
        self.assertFalse(marker_path.exists())

    def test_previous_schema_done_marker_is_requeued(self):
        video = self.make_video_placeholder("clip.mp4")
        spec = ray_batch.InputSpec(video.resolve(), Path("clip.mp4"))
        job = ray_batch.build_jobs([spec], self.config())[0]
        marker_path = ray_batch._state_path(self.output_root, "done", job.job_id)
        ray_batch.atomic_write_json(
            marker_path,
            {
                "schema_version": ray_batch.SCHEMA_VERSION - 1,
                "job_id": job.job_id,
            },
        )

        pending, completed, failed = ray_batch.classify_jobs(
            [job], self.output_root, self.output_root, retry_failed=False
        )

        self.assertEqual(pending, [job])
        self.assertEqual(completed, [])
        self.assertEqual(failed, [])
        self.assertFalse(marker_path.exists())

    def test_failed_marker_requires_explicit_retry(self):
        video = self.make_video_placeholder("clip.mp4")
        spec = ray_batch.InputSpec(video.resolve(), Path("clip.mp4"))
        job = ray_batch.build_jobs([spec], self.config())[0]
        failed_path = ray_batch._state_path(self.output_root, "failed", job.job_id)
        ray_batch.atomic_write_json(failed_path, {"job_id": job.job_id})

        pending, _, failed = ray_batch.classify_jobs(
            [job], self.output_root, self.output_root, retry_failed=False
        )
        retried, _, _ = ray_batch.classify_jobs(
            [job], self.output_root, self.output_root, retry_failed=True
        )

        self.assertEqual(pending, [])
        self.assertEqual(failed, [job])
        self.assertEqual(retried, [job])

    def test_real_video_validation_accepts_video_and_rejects_corruption(self):
        video = self.root / "tiny.mp4"
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (16, 16)
        )
        self.assertTrue(writer.isOpened())
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
        writer.write(np.full((16, 16, 3), 255, dtype=np.uint8))
        writer.release()

        valid, reason = ray_batch.validate_artifact(video, "video")
        self.assertTrue(valid, reason)

        corrupt = self.root / "corrupt.mp4"
        corrupt.write_bytes(b"not a video")
        valid, _ = ray_batch.validate_artifact(corrupt, "video")
        self.assertFalse(valid)

    def test_normalcrafter_to_udn_conversion(self):
        source = np.array(
            [
                [0.0, -2.0, 0.0],
                [0.0, 0.0, -3.0],
                [3.0, 4.0, 0.0],
                [np.nan, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        converted, valid = convert_normalcrafter_to_udn(source)

        np.testing.assert_allclose(converted[0], [0.0, -1.0, 0.0])
        np.testing.assert_allclose(converted[1], [0.0, 0.0, -1.0])
        np.testing.assert_allclose(converted[2], [0.6, 0.8, 0.0])
        np.testing.assert_allclose(converted[3:], [[0, 0, 1], [0, 0, 1]])
        np.testing.assert_allclose(
            np.linalg.norm(converted, axis=-1), np.ones(len(converted)), atol=1e-6
        )
        self.assertLess(converted[1, 2], 0.0)
        np.testing.assert_array_equal(valid, [True, True, True, False, False])

    def test_flip_y_switch_controls_ground_color(self):
        ground = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)

        default, _ = convert_normalcrafter_to_udn(ground)
        flipped, _ = convert_normalcrafter_to_udn(ground, flip_y=True)
        default_rgb = default * 0.5 + 0.5
        flipped_rgb = flipped * 0.5 + 0.5

        np.testing.assert_allclose(default[0, 0], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(default_rgb[0, 0], [0.5, 1.0, 0.5])
        np.testing.assert_allclose(flipped[0, 0], [0.0, -1.0, 0.0])
        np.testing.assert_allclose(flipped_rgb[0, 0], [0.5, 0.0, 0.5])

    def test_conversion_preserves_temporal_continuity_near_zero_z(self):
        z = np.array([0.02, -0.02] * 6, dtype=np.float32)
        x = np.sqrt(1.0 - z * z)
        source = np.zeros((len(z), 4, 5, 3), dtype=np.float32)
        source[..., 0] = x[:, None, None]
        source[..., 2] = z[:, None, None]

        converted, valid = convert_normalcrafter_to_udn(source)

        source_dots = np.sum(source[1:] * source[:-1], axis=-1)
        converted_dots = np.sum(converted[1:] * converted[:-1], axis=-1)
        near_tangent = (np.abs(source[1:, ..., 2]) < 0.05) & (
            np.abs(source[:-1, ..., 2]) < 0.05
        )
        ground = np.zeros(source.shape[1:3], dtype=np.bool_)
        ground[source.shape[1] // 2 :, :] = True
        source_flip_ratio = np.mean(source_dots[near_tangent] < -0.5)
        converted_flip_ratio = np.mean(converted_dots[near_tangent] < -0.5)
        source_ground_flip_ratio = np.mean(source_dots[:, ground] < -0.5)
        converted_ground_flip_ratio = np.mean(converted_dots[:, ground] < -0.5)

        self.assertTrue(np.all(valid))
        self.assertEqual(source_flip_ratio, 0.0)
        self.assertEqual(converted_flip_ratio, source_flip_ratio)
        self.assertEqual(converted_ground_flip_ratio, source_ground_flip_ratio)

        legacy = converted.copy()
        legacy[legacy[..., 2] < 0.0] *= -1.0
        legacy_dots = np.sum(legacy[1:] * legacy[:-1], axis=-1)
        self.assertGreater(np.mean(legacy_dots[near_tangent] < -0.5), 0.9)

    def test_udn_rgb_decoder_preserves_signed_z(self):
        normals = np.array(
            [
                [0.8, 0.0, -0.6],
                [0.8, 0.0, 0.6],
                [1.0, 0.0, -0.02],
            ],
            dtype=np.float32,
        )
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        rgb_u8 = np.rint(np.clip(normals * 0.5 + 0.5, 0.0, 1.0) * 255).astype(np.uint8)

        decoded, valid = decode_udn_rgb(rgb_u8)

        self.assertTrue(np.all(valid))
        self.assertLess(decoded[0, 2], 0.0)
        self.assertGreater(decoded[1, 2], 0.0)
        self.assertLess(decoded[2, 2], 0.0)
        np.testing.assert_allclose(
            np.linalg.norm(decoded, axis=-1), np.ones(len(decoded)), atol=1e-6
        )

    def test_udn_npz_fast_validation_reads_standard_metadata(self):
        path = self.root / "normal_udn.npz"
        normals = np.zeros((2, 4, 6, 3), dtype=np.float32)
        normals[..., 2] = 1.0
        normals[0, ..., 2] = -1.0
        np.savez_compressed(
            path,
            normal=normals,
            valid_mask=np.ones(normals.shape[:-1], dtype=np.bool_),
            standard=np.array("UDN-v2"),
            conformance=np.array(True),
            flip_y=np.array(False),
            coordinate_transform=np.array("identity"),
            signed_z=np.array(True),
            hemisphere_canonicalization=np.array("none"),
            coordinate_system=np.array(
                "right-handed-camera-x-right-y-up-z-toward-camera"
            ),
        )

        valid, reason = ray_batch.validate_artifact(path, "udn_npz")

        self.assertTrue(valid, reason)

    def test_udn_manifest_validation(self):
        path = self.root / "manifest.json"
        path.write_text(
            json.dumps(
                {
                    "standard": "UDN-v2",
                    "normal_conversion": {
                        "flip_y": False,
                        "coordinate_transform": "identity",
                    },
                    "normal_data": {
                        "dtype": "float32",
                        "npz_key": "normal",
                        "unit_length": True,
                        "conformance": True,
                        "flip_y": False,
                        "signed_z": True,
                        "hemisphere_canonicalization": "none",
                    },
                }
            ),
            encoding="utf-8",
        )

        valid, reason = ray_batch.validate_artifact(path, "udn_manifest")

        self.assertTrue(valid, reason)

    def test_y_flipped_manifest_is_valid_but_nonconformant(self):
        path = self.root / "y_flipped_manifest.json"
        path.write_text(
            json.dumps(
                {
                    "standard": "UDN-v2",
                    "conformance": False,
                    "normal_conversion": {
                        "flip_y": True,
                        "coordinate_transform": "diag(1,-1,1)",
                    },
                    "normal_data": {
                        "dtype": "float32",
                        "npz_key": "normal",
                        "unit_length": True,
                        "conformance": False,
                        "flip_y": True,
                        "signed_z": True,
                        "hemisphere_canonicalization": "none",
                    },
                }
            ),
            encoding="utf-8",
        )

        valid, reason = ray_batch.validate_artifact(path, "udn_manifest")

        self.assertTrue(valid, reason)

    def test_legacy_positive_z_manifest_is_rejected(self):
        path = self.root / "legacy_manifest.json"
        path.write_text(
            json.dumps(
                {
                    "standard": "UDN-v1",
                    "normal_data": {
                        "dtype": "float32",
                        "npz_key": "normal",
                        "unit_length": True,
                        "positive_z_hemisphere": True,
                    },
                }
            ),
            encoding="utf-8",
        )

        valid, _ = ray_batch.validate_artifact(path, "udn_manifest")

        self.assertFalse(valid)


if __name__ == "__main__":
    unittest.main()
