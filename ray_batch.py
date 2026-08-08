#!/usr/bin/env python3
"""Lightweight, resumable multi-GPU batch inference for NormalCrafter."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import gc
import hashlib
import json
import os
import shutil
import sys
import traceback
import uuid
import zipfile
from collections import deque
from pathlib import Path
from typing import Any, Iterator, Sequence


SCHEMA_VERSION = 6
UDN_STANDARDS = {"udn-v1", "udn-v2"}
VIDEO_EXTENSIONS = {
    ".avi",
    ".m2ts",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".mts",
    ".webm",
}


@dataclasses.dataclass(frozen=True)
class InputSpec:
    path: Path
    relative_path: Path


@dataclasses.dataclass(frozen=True)
class Job:
    job_id: str
    input_path: Path
    relative_path: Path
    output_files: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "input_path": str(self.input_path),
            "relative_path": self.relative_path.as_posix(),
            "output_files": [
                {"kind": kind, "path": path} for kind, path in self.output_files
            ],
        }


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def compute_job_id(input_path: Path, inference_config: dict[str, Any]) -> str:
    resolved = input_path.expanduser().resolve(strict=True)
    stat = resolved.stat()
    identity = {
        "schema_version": SCHEMA_VERSION,
        "input": {
            "path": str(resolved),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "inference": inference_config,
    }
    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()


def _is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def _is_below(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def discover_inputs(
    input_paths: Sequence[str],
    input_lists: Sequence[str],
    output_root: Path,
) -> list[InputSpec]:
    roots: list[Path] = []
    for raw_path in input_paths:
        roots.append(Path(raw_path).expanduser())

    for raw_list in input_lists:
        list_path = Path(raw_list).expanduser().resolve(strict=True)
        for raw_line in list_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            listed = Path(line).expanduser()
            if not listed.is_absolute():
                listed = list_path.parent / listed
            roots.append(listed)

    output_root = output_root.expanduser().resolve()
    discovered: dict[Path, InputSpec] = {}
    for root in roots:
        resolved_root = root.resolve(strict=True)
        if resolved_root.is_file():
            candidates = [(resolved_root, Path(resolved_root.name))]
        elif resolved_root.is_dir():
            candidates = (
                (path, path.relative_to(resolved_root))
                for path in sorted(resolved_root.rglob("*"))
            )
        else:
            raise ValueError(f"Input is neither a file nor directory: {resolved_root}")

        for path, relative_path in candidates:
            resolved_path = path.resolve()
            if _is_below(resolved_path, output_root) or not _is_video(resolved_path):
                continue
            discovered.setdefault(
                resolved_path,
                InputSpec(path=resolved_path, relative_path=relative_path),
            )

    return sorted(discovered.values(), key=lambda item: str(item.path))


def build_jobs(
    inputs: Sequence[InputSpec],
    job_identity_config: dict[str, Any],
) -> list[Job]:
    jobs: list[Job] = []
    for item in inputs:
        job_id = compute_job_id(item.path, job_identity_config)
        parent = item.relative_path.parent
        prefix = f"{item.relative_path.stem}_{job_id[:10]}"
        normal_standard = job_identity_config["inference"]["normal_standard"]
        output_mode = job_identity_config["batch"]["output_mode"]
        if output_mode == "normal-video-only":
            output_files = [
                ("video", item.relative_path.with_suffix(".mp4").as_posix())
            ]
            jobs.append(
                Job(
                    job_id=job_id,
                    input_path=item.path,
                    relative_path=item.relative_path,
                    output_files=tuple(output_files),
                )
            )
            continue
        output_files: list[tuple[str, str]] = [
            ("video", (parent / f"{prefix}_input.mp4").as_posix()),
            (
                "video",
                (
                    parent
                    / (
                        f"{prefix}_normal_udn_vis.mp4"
                        if normal_standard in UDN_STANDARDS
                        else f"{prefix}_vis.mp4"
                    )
                ).as_posix(),
            ),
        ]
        if job_identity_config["inference"]["save_npz"]:
            output_files.append(
                (
                    "udn_npz" if normal_standard in UDN_STANDARDS else "legacy_npz",
                    (
                        parent
                        / (
                            f"{prefix}_normal_udn.npz"
                            if normal_standard in UDN_STANDARDS
                            else f"{prefix}.npz"
                        )
                    ).as_posix(),
                )
            )
        if normal_standard in UDN_STANDARDS:
            output_files.append(
                ("udn_manifest", (parent / f"{prefix}_manifest.json").as_posix())
            )
        jobs.append(
            Job(
                job_id=job_id,
                input_path=item.path,
                relative_path=item.relative_path,
                output_files=tuple(output_files),
            )
        )
    return jobs


def _read_npy_header(archive: zipfile.ZipFile, member: str) -> tuple[Any, Any]:
    import numpy as np

    with archive.open(member) as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            shape, _, dtype = np.lib.format._read_array_header(handle, version)
    return shape, dtype


def validate_artifact(path: Path, kind: str) -> tuple[bool, str]:
    if not path.is_file() or path.stat().st_size == 0:
        return False, "missing or empty"
    try:
        if kind == "video":
            import decord

            reader = decord.VideoReader(str(path), num_threads=1)
            if len(reader) == 0:
                return False, "contains no frames"
            reader[0].asnumpy()
            if len(reader) > 1:
                reader[len(reader) - 1].asnumpy()
        elif kind in {"legacy_npz", "udn_npz"}:
            import numpy as np

            key = "normal" if kind == "udn_npz" else "depth"
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                if f"{key}.npy" not in names:
                    return False, f"missing {key} array"
                shape, dtype = _read_npy_header(archive, f"{key}.npy")
                if len(shape) != 4 or not all(size > 0 for size in shape):
                    return False, f"invalid {key} shape: {shape}"
                if shape[-1] != 3:
                    return False, f"invalid normal channel count: {shape}"
                if kind == "udn_npz":
                    if dtype != np.dtype(np.float32):
                        return False, f"UDN normal dtype is {dtype}, expected float32"
                    if "valid_mask.npy" not in names:
                        return False, "missing valid_mask array"
                    mask_shape, mask_dtype = _read_npy_header(archive, "valid_mask.npy")
                    if mask_shape != shape[:-1] or mask_dtype != np.dtype(np.bool_):
                        return False, "valid_mask shape or dtype mismatch"
            if kind == "udn_npz":
                with np.load(path, allow_pickle=False) as data:
                    if data["standard"].item() != "UDN-v2":
                        return False, "missing UDN-v2 standard metadata"
                    if not bool(data["signed_z"].item()):
                        return False, "UDN normal does not preserve signed Z"
                    if data["hemisphere_canonicalization"].item() != "none":
                        return False, "UDN normal uses hemisphere canonicalization"
                    flip_y = bool(data["flip_y"].item())
                    conformance = bool(data["conformance"].item())
                    expected_transform = "diag(1,-1,1)" if flip_y else "identity"
                    if data["coordinate_transform"].item() != expected_transform:
                        return False, "Y transform metadata mismatch"
                    if conformance == flip_y:
                        return False, "UDN conformance does not match Y transform"
                    coordinates = data["coordinate_system"].item()
                    expected_coordinates = (
                        "legacy-x-right-y-reflected-z-toward-camera"
                        if flip_y
                        else "right-handed-camera-x-right-y-up-z-toward-camera"
                    )
                    if coordinates != expected_coordinates:
                        return False, "coordinate system metadata mismatch"
        elif kind == "udn_manifest":
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("standard") != "UDN-v2":
                return False, "manifest standard is not UDN-v2"
            normal_data = manifest.get("normal_data", {})
            flip_y = normal_data.get("flip_y")
            if (
                normal_data.get("dtype") != "float32"
                or normal_data.get("npz_key") != "normal"
                or not normal_data.get("unit_length")
                or normal_data.get("signed_z") is not True
                or normal_data.get("hemisphere_canonicalization") != "none"
                or flip_y not in {True, False}
                or normal_data.get("conformance") != (not flip_y)
            ):
                return False, "manifest normal metadata is incomplete"
            conversion = manifest.get("normal_conversion", {})
            expected_transform = "diag(1,-1,1)" if flip_y else "identity"
            if (
                conversion.get("flip_y") is not flip_y
                or conversion.get("coordinate_transform") != expected_transform
            ):
                return False, "manifest Y transform metadata mismatch"
        else:
            return False, f"unknown artifact kind: {kind}"
    except Exception as exc:
        return False, f"validation failed: {exc}"
    return True, "ok"


def _state_path(output_root: Path, state: str, job_id: str) -> Path:
    return output_root / f".{state}" / f"{job_id}.json"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def valid_done_marker(job: Job, output_root: Path, state_root: Path) -> bool:
    marker_path = _state_path(state_root, "done", job.job_id)
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("job_id") != job.job_id
    ):
        return False
    expected = [path for _, path in job.output_files]
    if marker.get("outputs") != expected:
        return False
    expected_sizes = marker.get("output_sizes")
    expected_mtimes = marker.get("output_mtime_ns")
    if not isinstance(expected_sizes, dict) or not isinstance(expected_mtimes, dict):
        return False
    for kind, relative_path in job.output_files:
        artifact_path = output_root / relative_path
        if not artifact_path.is_file():
            return False
        stat = artifact_path.stat()
        if expected_sizes.get(relative_path) != stat.st_size:
            return False
        if expected_mtimes.get(relative_path) != stat.st_mtime_ns:
            return False
        valid, _ = validate_artifact(artifact_path, kind)
        if not valid:
            return False
    return True


def classify_jobs(
    jobs: Sequence[Job],
    output_root: Path,
    state_root: Path,
    retry_failed: bool,
) -> tuple[list[Job], list[Job], list[Job]]:
    pending: list[Job] = []
    completed: list[Job] = []
    failed: list[Job] = []
    for job in jobs:
        done_path = _state_path(state_root, "done", job.job_id)
        if valid_done_marker(job, output_root, state_root):
            completed.append(job)
            continue
        done_path.unlink(missing_ok=True)
        failed_path = _state_path(state_root, "failed", job.job_id)
        if failed_path.exists() and not retry_failed:
            failed.append(job)
            continue
        pending.append(job)
    return pending, completed, failed


@contextlib.contextmanager
def output_lock(output_root: Path) -> Iterator[None]:
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".ray_batch.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another batch driver holds the output lock: {lock_path}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def create_worker_class(ray: Any) -> Any:
    @ray.remote(num_gpus=1, max_restarts=0)
    class InferenceWorker:
        def __init__(
            self,
            repository_root: str,
            output_root: str,
            state_root: str,
            model_config: dict[str, Any],
            inference_config: dict[str, Any],
        ) -> None:
            if repository_root not in sys.path:
                sys.path.insert(0, repository_root)
            from run import DepthCrafterDemo

            self.output_root = Path(output_root)
            self.state_root = Path(state_root)
            self.inference_config = inference_config
            self.gpu_ids = ray.get_gpu_ids()
            self.visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            self.demo = DepthCrafterDemo(**model_config)

        def process(self, raw_job: dict[str, Any], attempt: int) -> dict[str, Any]:
            job_id = raw_job["job_id"]
            input_path = Path(raw_job["input_path"])
            partial_root = self.state_root / ".partial" / job_id
            try:
                shutil.rmtree(partial_root, ignore_errors=True)
                partial_root.mkdir(parents=True, exist_ok=True)
                generated = self.demo.infer(
                    str(input_path),
                    save_folder=str(partial_root),
                    **self.inference_config,
                )
                output_files = raw_job["output_files"]
                if len(generated) != len(output_files):
                    raise RuntimeError(
                        "Generated artifact count does not match the job: "
                        f"{len(generated)} != {len(output_files)}"
                    )
                partial_files = [
                    (expected["kind"], Path(path))
                    for expected, path in zip(output_files, generated)
                ]

                for kind, path in partial_files:
                    valid, reason = validate_artifact(path, kind)
                    if not valid:
                        raise RuntimeError(
                            f"Invalid generated artifact {path}: {reason}"
                        )

                final_paths: list[str] = []
                for (kind, source), expected in zip(partial_files, output_files):
                    if kind != expected["kind"]:
                        raise RuntimeError(
                            "Generated artifact type does not match the job"
                        )
                    destination = self.output_root / expected["path"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(source, destination)
                    final_paths.append(expected["path"])

                manifest_outputs = [
                    expected
                    for expected in output_files
                    if expected["kind"] == "udn_manifest"
                ]
                if manifest_outputs:
                    manifest_path = self.output_root / manifest_outputs[0]["path"]
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["artifacts"] = {
                        "input_preview": next(
                            item["path"]
                            for item in output_files
                            if item["kind"] == "video" and "_input.mp4" in item["path"]
                        ),
                        "normal_preview": next(
                            item["path"]
                            for item in output_files
                            if item["kind"] == "video" and "_vis.mp4" in item["path"]
                        ),
                        "canonical_normal": next(
                            item["path"]
                            for item in output_files
                            if item["kind"] == "udn_npz"
                        ),
                    }
                    atomic_write_json(manifest_path, manifest)

                for expected in output_files:
                    valid, reason = validate_artifact(
                        self.output_root / expected["path"], expected["kind"]
                    )
                    if not valid:
                        raise RuntimeError(
                            f"Invalid finalized artifact {expected['path']}: {reason}"
                        )

                marker = {
                    "schema_version": SCHEMA_VERSION,
                    "job_id": job_id,
                    "input": str(input_path),
                    "outputs": final_paths,
                    "output_sizes": {
                        expected["path"]: (self.output_root / expected["path"])
                        .stat()
                        .st_size
                        for expected in output_files
                    },
                    "output_mtime_ns": {
                        expected["path"]: (self.output_root / expected["path"])
                        .stat()
                        .st_mtime_ns
                        for expected in output_files
                    },
                    "attempt": attempt,
                    "gpu_ids": self.gpu_ids,
                    "cuda_visible_devices": self.visible_devices,
                }
                atomic_write_json(_state_path(self.state_root, "done", job_id), marker)
                _state_path(self.state_root, "failed", job_id).unlink(missing_ok=True)
                shutil.rmtree(partial_root, ignore_errors=True)
                return {"ok": True, **marker}
            except Exception as exc:
                return {
                    "ok": False,
                    "job_id": job_id,
                    "input": str(input_path),
                    "attempt": attempt,
                    "gpu_ids": self.gpu_ids,
                    "cuda_visible_devices": self.visible_devices,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            finally:
                gc.collect()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass

    return InferenceWorker


def run_ray_batch(
    ray: Any,
    jobs: Sequence[Job],
    output_root: Path,
    state_root: Path,
    repository_root: Path,
    model_config: dict[str, Any],
    inference_config: dict[str, Any],
    workers: int,
    retries: int,
) -> tuple[int, int]:
    worker_class = create_worker_class(ray)
    actors = [
        worker_class.remote(
            str(repository_root),
            str(output_root),
            str(state_root),
            model_config,
            inference_config,
        )
        for _ in range(workers)
    ]
    pending = deque((job, 1) for job in jobs)
    idle = deque(range(workers))
    in_flight: dict[Any, tuple[int, Job, int]] = {}
    succeeded = 0
    failed = 0
    max_attempts = 1 + retries

    while pending or in_flight:
        while pending and idle:
            worker_index = idle.popleft()
            job, attempt = pending.popleft()
            ref = actors[worker_index].process.remote(job.to_dict(), attempt)
            in_flight[ref] = (worker_index, job, attempt)
            print(
                f"[start {attempt}/{max_attempts}] {job.input_path} "
                f"(worker {worker_index})",
                flush=True,
            )

        ready, _ = ray.wait(list(in_flight), num_returns=1)
        ref = ready[0]
        worker_index, job, attempt = in_flight.pop(ref)
        actor_failed = False
        try:
            result = ray.get(ref)
        except Exception as exc:
            actor_failed = True
            result = {
                "ok": False,
                "job_id": job.job_id,
                "input": str(job.input_path),
                "attempt": attempt,
                "error": f"Ray worker failure: {exc}",
                "traceback": traceback.format_exc(),
            }

        if result["ok"]:
            succeeded += 1
            print(
                f"[done] {job.input_path} gpu={result.get('gpu_ids', [])} "
                f"visible={result.get('cuda_visible_devices', '')}",
                flush=True,
            )
        elif attempt < max_attempts:
            print(
                f"[retry {attempt + 1}/{max_attempts}] {job.input_path}: "
                f"{result['error']}",
                file=sys.stderr,
                flush=True,
            )
            pending.append((job, attempt + 1))
        else:
            failed += 1
            atomic_write_json(
                _state_path(state_root, "failed", job.job_id),
                {
                    "schema_version": SCHEMA_VERSION,
                    **result,
                    "max_attempts": max_attempts,
                },
            )
            print(
                f"[failed] {job.input_path}: {result['error']}",
                file=sys.stderr,
                flush=True,
            )

        if actor_failed:
            actors[worker_index] = worker_class.remote(
                str(repository_root),
                str(output_root),
                str(state_root),
                model_config,
                inference_config,
            )
        idle.append(worker_index)

    return succeeded, failed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable one-process-per-GPU batch inference for NormalCrafter."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Input video or directory; may be repeated.",
    )
    parser.add_argument(
        "--input-list",
        action="append",
        default=[],
        help="Text file containing one input path per line; may be repeated.",
    )
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument(
        "--state-dir",
        help=(
            "External resume-state directory. Normal-video-only mode defaults to "
            "a hidden sibling of --output."
        ),
    )
    parser.add_argument(
        "--output-mode",
        choices=("full", "normal-video-only"),
        default="full",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="GPU workers (default: all GPUs visible to Ray).",
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ray-temp-dir", default="/data/bc/.cache/ray/normalcrafter")
    parser.add_argument("--hf-home", default="/data/huggingface")
    parser.add_argument("--offline", action="store_true")

    parser.add_argument("--unet-path", default="Yanrui95/NormalCrafter")
    parser.add_argument(
        "--pre-train-path",
        default="stabilityai/stable-video-diffusion-img2vid-xt",
    )
    parser.add_argument(
        "--cpu-offload", choices=("model", "sequential", "none"), default="model"
    )
    parser.add_argument("--process-length", type=int, default=-1)
    parser.add_argument("--target-fps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=14)
    parser.add_argument("--time-step-size", type=int, default=10)
    parser.add_argument("--decode-chunk-size", type=int, default=7)
    parser.add_argument("--max-res", type=int, default=1024)
    parser.add_argument("--dataset", default="open")
    parser.add_argument("--save-npz", action="store_true")
    parser.add_argument(
        "--flip-y",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply a fixed Ny=-Ny reflection. Disabled by default so the "
            "NormalCrafter ground remains green."
        ),
    )
    parser.add_argument(
        "--normal-standard",
        choices=("normalcrafter", "udn-v1", "udn-v2"),
        default="normalcrafter",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.normal_standard == "udn-v1":
        print(
            "Warning: --normal-standard udn-v1 is deprecated; using udn-v2.",
            file=sys.stderr,
            flush=True,
        )
        args.normal_standard = "udn-v2"
    if not args.input and not args.input_list:
        raise SystemExit("At least one --input or --input-list is required")
    if args.workers < 0:
        raise SystemExit("--workers must be zero or greater")
    if args.retries < 0:
        raise SystemExit("--retries must be zero or greater")
    if (
        args.output_mode == "full"
        and args.normal_standard == "udn-v2"
        and not args.save_npz
    ):
        raise SystemExit("--normal-standard udn-v2 requires --save-npz")
    if args.output_mode == "normal-video-only" and args.save_npz:
        raise SystemExit(
            "--output-mode normal-video-only cannot be used with --save-npz"
        )
    if args.output_mode == "normal-video-only" and args.normal_standard != "udn-v2":
        raise SystemExit(
            "--output-mode normal-video-only requires --normal-standard udn-v2"
        )

    repository_root = Path(__file__).resolve().parent
    output_root = Path(args.output).expanduser().resolve()
    if args.state_dir:
        state_root = Path(args.state_dir).expanduser().resolve()
    elif args.output_mode == "normal-video-only":
        state_root = output_root.parent / f".{output_root.name}.state"
    else:
        state_root = output_root
    if args.output_mode == "normal-video-only" and state_root == output_root:
        raise SystemExit("normal-video-only mode requires --state-dir outside --output")
    os.environ["HF_HOME"] = str(Path(args.hf_home).expanduser().resolve())
    os.environ["PATH"] = f"{sys.prefix}/bin:{os.environ.get('PATH', '')}"
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    model_config = {
        "unet_path": args.unet_path,
        "pre_train_path": args.pre_train_path,
        "cpu_offload": None if args.cpu_offload == "none" else args.cpu_offload,
    }
    inference_config = {
        "window_size": args.window_size,
        "time_step_size": args.time_step_size,
        "process_length": args.process_length,
        "decode_chunk_size": args.decode_chunk_size,
        "max_res": args.max_res,
        "dataset": args.dataset,
        "target_fps": args.target_fps,
        "seed": args.seed,
        "save_npz": args.save_npz,
        "normal_standard": args.normal_standard,
        "flip_y": args.flip_y,
        "save_input_video": args.output_mode != "normal-video-only",
        "save_manifest": args.output_mode != "normal-video-only",
    }
    job_identity_config = {
        "model": model_config,
        "inference": inference_config,
        "batch": {"output_mode": args.output_mode},
    }

    inputs = discover_inputs(args.input, args.input_list, output_root)
    if not inputs:
        raise SystemExit("No supported input videos found")
    jobs = build_jobs(inputs, job_identity_config)

    with output_lock(state_root):
        if args.output_mode == "normal-video-only":
            atomic_write_json(
                state_root / "dataset_manifest.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "standard": "UDN-v2",
                    "conformance": not args.flip_y,
                    "compliance": (
                        "Legacy" if args.flip_y else "Transport-approximate"
                    ),
                    "source_root": [
                        str(Path(path).expanduser().resolve()) for path in args.input
                    ],
                    "output_root": str(output_root),
                    "job_count": len(jobs),
                    "normal_conversion": {
                        "source_coordinates": "right-handed-camera-x-right-y-up-z-toward-camera",
                        "target_coordinates": (
                            "legacy-x-right-y-reflected-z-toward-camera"
                            if args.flip_y
                            else "right-handed-camera-x-right-y-up-z-toward-camera"
                        ),
                        "coordinate_transform": (
                            "diag(1,-1,1)" if args.flip_y else "identity"
                        ),
                        "flip_y": args.flip_y,
                        "operations": (["reflect-y"] if args.flip_y else [])
                        + [
                            "normalize-float32",
                            "signed-z-preservation",
                            "invalid-to-0-0-1",
                            "normal-rgb-mapping-without-eotf-oetf",
                        ],
                    },
                    "normal_data": {
                        "unit_length": True,
                        "conformance": not args.flip_y,
                        "flip_y": args.flip_y,
                        "signed_z": True,
                        "hemisphere_canonicalization": "none",
                    },
                    "transport": {
                        "container": "MP4",
                        "codec": "H.264",
                        "pixel_format": "yuv420p",
                        "bit_depth": 8,
                        "color_matrix": "BT.709",
                        "color_range": "Limited",
                        "coordinate_transform": (
                            "diag(1,-1,1)" if args.flip_y else "identity"
                        ),
                        "decode": "normalize(2*RGB-1)",
                        "status": "lossy preview/data transport; not canonical float",
                    },
                    "inference": inference_config,
                    "model": model_config,
                },
            )
        pending, completed, prior_failed = classify_jobs(
            jobs, output_root, state_root, args.retry_failed
        )
        print(
            f"Discovered {len(jobs)} job(s): {len(pending)} pending, "
            f"{len(completed)} completed, {len(prior_failed)} failed/skipped.",
            flush=True,
        )
        if args.dry_run:
            for job in pending:
                print(f"[pending] {job.input_path} -> {job.output_files[-1][1]}")
            return 0
        if not pending:
            return 1 if prior_failed else 0

        import ray

        ray_temp_dir = Path(args.ray_temp_dir).expanduser().resolve()
        ray_temp_dir.mkdir(parents=True, exist_ok=True)
        ray.init(_temp_dir=str(ray_temp_dir), include_dashboard=False)
        try:
            gpu_count = int(ray.cluster_resources().get("GPU", 0))
            if gpu_count < 1:
                raise RuntimeError("Ray detected no GPU resources")
            workers = args.workers or gpu_count
            if workers > gpu_count:
                raise RuntimeError(
                    f"Requested {workers} workers, but Ray detected {gpu_count} GPUs"
                )
            print(
                f"Starting {workers} worker(s) on {gpu_count} Ray GPU resource(s).",
                flush=True,
            )
            succeeded, failed = run_ray_batch(
                ray=ray,
                jobs=pending,
                output_root=output_root,
                state_root=state_root,
                repository_root=repository_root,
                model_config=model_config,
                inference_config=inference_config,
                workers=workers,
                retries=args.retries,
            )
        finally:
            ray.shutdown()

        print(
            f"Batch finished: {succeeded} succeeded, {failed} failed, "
            f"{len(completed)} previously completed.",
            flush=True,
        )
        return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
