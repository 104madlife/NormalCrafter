import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from pipeline_adapters import run_normal_from_index as adapter


class PipelineAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def make_video(self, name: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (16, 16)
        )
        self.assertTrue(writer.isOpened())
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
        writer.write(np.full((16, 16, 3), 255, dtype=np.uint8))
        writer.release()
        return path

    def make_items(self, count: int = 2) -> tuple[list[dict], Path]:
        rows = []
        for index in range(count):
            item_id = f"task__GT__slice_{index:06d}"
            rows.append(
                {
                    "schema_version": "dataset_item.v1",
                    "item_id": item_id,
                    "channel": "GT",
                    "video_key": f"slice/{item_id}.mp4",
                    "local_video_path": str(self.make_video(f"inputs/{item_id}.mp4")),
                }
            )
        path = self.root / "prepared_items.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return rows, path

    def runtime_input(self, prepared_path: Path, *, min_gpus: int = 2) -> dict:
        work_dir = self.root / "work"
        output_dir = self.root / "outputs"
        return {
            "task_id": "normal_smoke",
            "stage": "normalcrafter_v1",
            "task": {
                "task_id": "normal_smoke",
                "batch": {"run_id": "normal_smoke"},
                "artifacts": {
                    "build_normal_index_v1": {
                        "prepared_items_local_path": str(prepared_path),
                    }
                },
            },
            "parameters": {
                "index_stage": "build_normal_index_v1",
                "output_prefix": "smoke/normalcrafter/normal_smoke/",
                "output_mode": "normal-video-only",
                "normal_standard": "udn-v2",
                "workers": "auto",
                "min_ray_gpu_count": min_gpus,
                "offline": True,
            },
            "work_dir": str(work_dir),
            "output_dir": str(output_dir),
        }

    def test_read_prepared_gt_items(self):
        expected, path = self.make_items()
        runtime_input = self.runtime_input(path)

        items, resolved_path = adapter.read_prepared_items(
            runtime_input, work_dir=self.root / "work"
        )

        self.assertEqual(items, expected)
        self.assertEqual(resolved_path, path)

    def test_non_gt_item_is_rejected(self):
        rows, path = self.make_items(1)
        rows[0]["channel"] = "normal"
        path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(adapter.AdapterFailure, "channel=GT"):
            adapter.read_prepared_items(
                self.runtime_input(path), work_dir=self.root / "work"
            )

    def test_gpu_count_requires_multi_gpu_allocation(self):
        params = {"workers": "auto", "min_ray_gpu_count": 2}
        with mock.patch.dict(
            os.environ,
            {
                "BOS_PIPELINE_ALLOCATED_GPU_COUNT": "1",
                "CUDA_VISIBLE_DEVICES": "4",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(adapter.AdapterFailure, "at least 2"):
                adapter.resolve_allocated_gpu_count(params)

        with mock.patch.dict(
            os.environ,
            {
                "BOS_PIPELINE_ALLOCATED_GPU_COUNT": "2",
                "CUDA_VISIBLE_DEVICES": "4,7",
            },
            clear=False,
        ):
            self.assertEqual(adapter.resolve_allocated_gpu_count(params), (2, 2))

    def test_ray_command_uses_allocated_worker_count(self):
        command = adapter.build_ray_command(
            params={
                "output_mode": "normal-video-only",
                "normal_standard": "udn-v2",
                "offline": True,
                "flip_y": False,
            },
            input_list=self.root / "inputs.txt",
            output_root=self.root / "outputs",
            state_root=self.root / "state",
            workers=2,
        )

        self.assertEqual(command[command.index("--workers") + 1], "2")
        self.assertIn("--offline", command)
        self.assertIn("--no-flip-y", command)
        self.assertIn("--retry-failed", command)

    def test_aggregate_records_distinct_ray_gpu_ids(self):
        items, _ = self.make_items()
        output_root = self.root / "normal_outputs"
        state_root = self.root / "state"
        for index, item in enumerate(items):
            output = output_root / f"{adapter.safe_item_stem(item['item_id'])}.mp4"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"normal")
            adapter.write_json(
                state_root / ".done" / f"job-{index}.json",
                {
                    "input": item["local_video_path"],
                    "gpu_ids": [float(index)],
                    "cuda_visible_devices": str(index),
                },
            )

        result = adapter.aggregate_outputs(
            items=items,
            output_root=output_root,
            state_root=state_root,
            output_prefix="smoke/run/",
            validator=lambda _path, _kind: (True, "ok"),
        )

        self.assertEqual(result["success_count"], 2)
        self.assertEqual(result["failure_count"], 0)
        self.assertEqual(result["used_ray_gpu_ids"], ["0", "1"])

    def fake_ray_run(self, command: list[str], *, distinct_gpus: bool = True):
        input_list = Path(command[command.index("--input-list") + 1])
        output_root = Path(command[command.index("--output") + 1])
        state_root = Path(command[command.index("--state-dir") + 1])
        for index, line in enumerate(input_list.read_text().splitlines()):
            source = Path(line)
            destination = output_root / source.with_suffix(".mp4").name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            gpu_id = index if distinct_gpus else 0
            adapter.write_json(
                state_root / ".done" / f"job-{index}.json",
                {
                    "input": str(source),
                    "outputs": [destination.name],
                    "gpu_ids": [float(gpu_id)],
                    "cuda_visible_devices": str(gpu_id),
                },
            )
        return subprocess.CompletedProcess(command, 0)

    def run_main_with_fake_ray(self, *, distinct_gpus: bool) -> dict:
        _, prepared_path = self.make_items()
        runtime_input_path = self.root / "runtime_input.json"
        runtime_output_path = self.root / "runtime_output.json"
        adapter.write_json(runtime_input_path, self.runtime_input(prepared_path))

        def fake_run(command, **_kwargs):
            return self.fake_ray_run(command, distinct_gpus=distinct_gpus)

        with (
            mock.patch.dict(
                os.environ,
                {
                    "BOS_PIPELINE_ALLOCATED_GPU_COUNT": "2",
                    "CUDA_VISIBLE_DEVICES": "0,1",
                },
                clear=False,
            ),
            mock.patch.object(adapter.subprocess, "run", side_effect=fake_run),
        ):
            exit_code = adapter.main(
                [
                    "--input",
                    str(runtime_input_path),
                    "--output",
                    str(runtime_output_path),
                ]
            )
        self.assertEqual(exit_code, 0)
        return json.loads(runtime_output_path.read_text())

    def test_main_writes_done_for_two_gpu_coverage(self):
        result = self.run_main_with_fake_ray(distinct_gpus=True)

        self.assertEqual(result["status"], "done")
        self.assertTrue(result["artifacts"]["multi_gpu_coverage_ok"])
        self.assertEqual(result["artifacts"]["used_ray_gpu_ids"], ["0", "1"])
        self.assertEqual(len(result["output_files"]), 5)

    def test_main_fails_when_ray_only_uses_one_gpu(self):
        result = self.run_main_with_fake_ray(distinct_gpus=False)

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["artifacts"]["multi_gpu_coverage_ok"])
        self.assertIn("expected at least 2", result["reason"])


if __name__ == "__main__":
    unittest.main()
