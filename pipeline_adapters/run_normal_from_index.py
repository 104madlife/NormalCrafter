#!/usr/bin/env python3
"""Run a prepared GT-video index through NormalCrafter's Ray batch driver."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import ray_batch  # noqa: E402


class AdapterFailure(Exception):
    """A failure that should be returned through the pipeline runtime protocol."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run NormalCrafter from a pipeline prepared-items index."
    )
    parser.add_argument("--input", required=True, help="Path to runtime_input.json.")
    parser.add_argument("--output", required=True, help="Path to runtime_output.json.")
    return parser


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def write_runtime_output(
    path: Path,
    *,
    status: str,
    artifacts: dict[str, Any] | None = None,
    output_files: list[dict[str, str]] | None = None,
    reason: str | None = None,
) -> None:
    write_json(
        path,
        {
            "status": status,
            "artifacts": artifacts or {},
            "output_files": output_files or [],
            "next_stage_override": None,
            "stop_pipeline": False,
            "reason": reason,
        },
    )


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_int(value: Any, *, name: str, default: int, minimum: int = 0) -> int:
    if value is None or str(value).strip() == "":
        parsed = default
    else:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise AdapterFailure(f"{name} must be an integer: {value!r}") from exc
    if parsed < minimum:
        raise AdapterFailure(f"{name} must be at least {minimum}: {parsed}")
    return parsed


def safe_item_stem(value: Any) -> str:
    text = str(value or "")
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return safe.strip("._") or "item"


def ensure_trailing_slash(value: str) -> str:
    normalized = value.strip().lstrip("/")
    if not normalized:
        return ""
    return normalized if normalized.endswith("/") else f"{normalized}/"


def run_id_from_input(runtime_input: dict[str, Any]) -> str:
    task = runtime_input.get("task") or {}
    batch = task.get("batch") or {}
    value = batch.get("run_id") or task.get("task_id") or runtime_input.get("task_id")
    if not value or not str(value).strip():
        raise AdapterFailure("Runtime input does not contain a run_id or task_id.")
    return str(value).strip()


def _index_artifacts(runtime_input: dict[str, Any], index_stage: str) -> dict[str, Any]:
    task = runtime_input.get("task") or {}
    artifacts = task.get("artifacts") or {}
    value = artifacts.get(index_stage) or {}
    if not isinstance(value, dict):
        raise AdapterFailure(f"task.artifacts.{index_stage} must be an object.")
    return value


def _download_prepared_items(key: str, destination: Path) -> None:
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from bos_gbuffer_pipeline import storage  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        raise AdapterFailure(
            "Prepared items are not available locally and the BOS pipeline package "
            "cannot be imported in the algorithm environment."
        ) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(storage.read_text(key), encoding="utf-8")


def read_prepared_items(
    runtime_input: dict[str, Any],
    *,
    work_dir: Path,
) -> tuple[list[dict[str, Any]], Path]:
    params = runtime_input.get("parameters") or {}
    if not isinstance(params, dict):
        params = {}
    index_stage = str(params.get("index_stage") or "build_normal_index_v1")
    artifacts = _index_artifacts(runtime_input, index_stage)
    local_value = params.get("prepared_items_local_path") or artifacts.get(
        "prepared_items_local_path"
    )
    key_value = params.get("prepared_items_key") or artifacts.get("prepared_items_key")

    if local_value and Path(str(local_value)).is_file():
        path = Path(str(local_value))
    elif key_value:
        path = work_dir / "prepared_items.jsonl"
        _download_prepared_items(str(key_value), path)
    else:
        raise AdapterFailure(
            f"Missing artifacts.{index_stage}.prepared_items_local_path/prepared_items_key."
        )

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AdapterFailure(
                f"Invalid prepared-items JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise AdapterFailure(
                f"Prepared item at {path}:{line_number} must be an object."
            )
        if not row.get("item_id"):
            raise AdapterFailure(
                f"Prepared item at {path}:{line_number} is missing item_id."
            )
        if str(row.get("channel") or "") != "GT":
            raise AdapterFailure(
                f"NormalCrafter requires channel=GT: item_id={row.get('item_id')}"
            )
        local_video = Path(str(row.get("local_video_path") or ""))
        if not local_video.is_file():
            raise AdapterFailure(
                f"Prepared GT video is missing: item_id={row.get('item_id')} "
                f"path={local_video}"
            )
        if "\n" in str(local_video) or "\r" in str(local_video):
            raise AdapterFailure(f"Prepared path contains a newline: {local_video}")
        rows.append(row)
    if not rows:
        raise AdapterFailure(f"Prepared item list is empty: {path}")
    return rows, path


def select_items(
    items: list[dict[str, Any]], params: dict[str, Any]
) -> list[dict[str, Any]]:
    limit_value = params.get("limit_items")
    if limit_value is None:
        return items
    limit = parse_int(limit_value, name="limit_items", default=len(items), minimum=1)
    return items[:limit]


def resolve_allocated_gpu_count(params: dict[str, Any]) -> tuple[int, int]:
    minimum = parse_int(
        params.get("min_ray_gpu_count"),
        name="min_ray_gpu_count",
        default=1,
        minimum=1,
    )
    explicit = params.get("workers")
    allocated_text = os.environ.get("BOS_PIPELINE_ALLOCATED_GPU_COUNT", "").strip()
    visible = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    if explicit is not None and str(explicit).strip().lower() != "auto":
        count = parse_int(explicit, name="workers", default=minimum, minimum=1)
    elif allocated_text:
        count = parse_int(
            allocated_text,
            name="BOS_PIPELINE_ALLOCATED_GPU_COUNT",
            default=minimum,
            minimum=1,
        )
    else:
        count = len(visible)
    if count < minimum:
        raise AdapterFailure(
            f"NormalCrafter requires at least {minimum} allocated GPUs, got {count}."
        )
    if visible and count > len(visible):
        raise AdapterFailure(
            f"Requested {count} Ray workers but CUDA_VISIBLE_DEVICES exposes "
            f"only {len(visible)} GPU(s)."
        )
    return count, minimum


def _append_value_arg(command: list[str], flag: str, value: Any) -> None:
    if value is not None and str(value).strip() != "":
        command.extend([flag, str(value)])


def build_ray_command(
    *,
    params: dict[str, Any],
    input_list: Path,
    output_root: Path,
    state_root: Path,
    workers: int,
) -> list[str]:
    output_mode = str(params.get("output_mode") or "normal-video-only")
    normal_standard = str(params.get("normal_standard") or "udn-v2")
    if output_mode != "normal-video-only":
        raise AdapterFailure(
            "normalcrafter_v1 currently supports output_mode=normal-video-only only."
        )
    if normal_standard != "udn-v2":
        raise AdapterFailure("normalcrafter_v1 requires normal_standard=udn-v2.")

    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "ray_batch.py"),
        "--input-list",
        str(input_list),
        "--output",
        str(output_root),
        "--state-dir",
        str(state_root),
        "--output-mode",
        output_mode,
        "--normal-standard",
        normal_standard,
        "--workers",
        str(workers),
        "--retries",
        str(parse_int(params.get("retries"), name="retries", default=2)),
        "--retry-failed",
    ]
    value_args = {
        "--ray-temp-dir": params.get("ray_temp_dir"),
        "--hf-home": params.get("hf_home"),
        "--unet-path": params.get("unet_path"),
        "--pre-train-path": params.get("pre_train_path"),
        "--cpu-offload": params.get("cpu_offload"),
        "--process-length": params.get("process_length"),
        "--target-fps": params.get("target_fps"),
        "--seed": params.get("seed"),
        "--window-size": params.get("window_size"),
        "--time-step-size": params.get("time_step_size"),
        "--decode-chunk-size": params.get("decode_chunk_size"),
        "--max-res": params.get("max_res"),
        "--dataset": params.get("dataset"),
    }
    for flag, value in value_args.items():
        _append_value_arg(command, flag, value)
    if parse_bool(params.get("offline"), default=True):
        command.append("--offline")
    if parse_bool(params.get("flip_y"), default=False):
        command.append("--flip-y")
    else:
        command.append("--no-flip-y")
    return command


def _gpu_token(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else str(number)


def read_ray_evidence(state_root: Path) -> dict[str, dict[str, Any]]:
    evidence: dict[str, tuple[int, dict[str, Any]]] = {}
    for path in (state_root / ".done").glob("*.json"):
        try:
            marker = json.loads(path.read_text(encoding="utf-8"))
            input_path = str(Path(str(marker["input"])).resolve())
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        mtime = path.stat().st_mtime_ns
        if input_path not in evidence or mtime > evidence[input_path][0]:
            evidence[input_path] = (mtime, marker)
    return {key: value[1] for key, value in evidence.items()}


def probe_video(path: Path) -> dict[str, Any]:
    """Read the frame-level properties used by the production output gate."""
    import decord

    reader = decord.VideoReader(str(path), num_threads=1)
    frame_count = len(reader)
    if frame_count <= 0:
        raise ValueError("contains no frames")
    first_frame = reader[0]
    height, width = first_frame.shape[:2]
    return {
        "frame_count": frame_count,
        "width": int(width),
        "height": int(height),
        "fps": float(reader.get_avg_fps()),
    }


def aggregate_outputs(
    *,
    items: list[dict[str, Any]],
    output_root: Path,
    state_root: Path,
    output_prefix: str,
    flip_y: bool = False,
    validator: Callable[[Path, str], tuple[bool, str]] = ray_batch.validate_artifact,
    video_probe: Callable[[Path], dict[str, Any]] = probe_video,
) -> dict[str, Any]:
    evidence = read_ray_evidence(state_root)
    rows: list[dict[str, Any]] = []
    success_files: list[tuple[Path, str]] = []
    used_ray_gpu_ids: set[str] = set()
    used_cuda_visible_devices: set[str] = set()
    seen_names: set[str] = set()

    for item in items:
        item_id = str(item["item_id"])
        input_filename = Path(str(item["local_video_path"])).with_suffix(".mp4").name
        if input_filename in seen_names:
            raise AdapterFailure(f"Duplicate prepared input filename: {input_filename}")
        seen_names.add(input_filename)
        local_output = output_root / input_filename
        output_filename = f"{safe_item_stem(item_id)}.mp4"
        output_key = f"{output_prefix}predicted_normal/{output_filename}"
        valid, validation_reason = validator(local_output, "video")
        input_video: dict[str, Any] | None = None
        output_video: dict[str, Any] | None = None
        if valid:
            try:
                input_video = video_probe(Path(str(item["local_video_path"])))
                output_video = video_probe(local_output)
                expected_frame_count = item.get("actual_frame_count")
                if expected_frame_count is not None and int(expected_frame_count) != int(
                    input_video["frame_count"]
                ):
                    valid = False
                    validation_reason = (
                        "prepared GT frame count does not match the source manifest: "
                        f"expected={expected_frame_count}, "
                        f"actual={input_video['frame_count']}"
                    )
                elif int(output_video["frame_count"]) != int(
                    input_video["frame_count"]
                ):
                    valid = False
                    validation_reason = (
                        "predicted-normal frame count does not match prepared GT: "
                        f"input={input_video['frame_count']}, "
                        f"output={output_video['frame_count']}"
                    )
            except Exception as exc:
                valid = False
                validation_reason = f"frame-count gate could not probe video: {exc}"
        marker = evidence.get(str(Path(str(item["local_video_path"])).resolve())) or {}
        ray_gpu_ids = [_gpu_token(value) for value in marker.get("gpu_ids") or []]
        cuda_visible = str(marker.get("cuda_visible_devices") or "").strip()
        used_ray_gpu_ids.update(ray_gpu_ids)
        if cuda_visible:
            used_cuda_visible_devices.add(cuda_visible)
        if valid:
            success_files.append((local_output, output_key))
            rows.append(
                {
                    **item,
                    "status": "done",
                    "predicted_normal_key": output_key,
                    "normal_standard": "UDN-v2",
                    "flip_y": flip_y,
                    "ray_gpu_ids": ray_gpu_ids,
                    "cuda_visible_devices": cuda_visible or None,
                    "input_frame_count": input_video["frame_count"],
                    "output_frame_count": output_video["frame_count"],
                    "input_fps": input_video["fps"],
                    "output_fps": output_video["fps"],
                }
            )
        else:
            rows.append(
                {
                    **item,
                    "status": "failed",
                    "error": f"Predicted normal output is invalid: {validation_reason}",
                    "retryable": True,
                    "ray_gpu_ids": ray_gpu_ids,
                    "cuda_visible_devices": cuda_visible or None,
                    "input_frame_count": (
                        input_video.get("frame_count") if input_video else None
                    ),
                    "output_frame_count": (
                        output_video.get("frame_count") if output_video else None
                    ),
                }
            )

    failed_rows = [row for row in rows if row["status"] == "failed"]
    return {
        "rows": rows,
        "failed_rows": failed_rows,
        "success_files": success_files,
        "success_count": len(rows) - len(failed_rows),
        "failure_count": len(failed_rows),
        "used_ray_gpu_ids": sorted(used_ray_gpu_ids),
        "used_cuda_visible_devices": sorted(used_cuda_visible_devices),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime_input_path = Path(args.input)
    runtime_output_path = Path(args.output)
    output_files: list[dict[str, str]] = []

    try:
        runtime_input = json.loads(runtime_input_path.read_text(encoding="utf-8"))
        params = runtime_input.get("parameters") or {}
        if not isinstance(params, dict):
            params = {}
        run_id = run_id_from_input(runtime_input)
        work_dir = Path(str(runtime_input["work_dir"]))
        runtime_output_dir = Path(str(runtime_input["output_dir"]))
        items, prepared_items_path = read_prepared_items(
            runtime_input, work_dir=work_dir
        )
        selected_items = select_items(items, params)
        workers, min_gpu_count = resolve_allocated_gpu_count(params)
        if len(selected_items) < min_gpu_count:
            raise AdapterFailure(
                f"A {min_gpu_count}-GPU coverage test requires at least "
                f"{min_gpu_count} prepared items, got {len(selected_items)}."
            )

        batch_root = runtime_output_dir / "normalcrafter_batch"
        normal_output_root = batch_root / "predicted_normal"
        state_root_value = params.get("state_root")
        state_root = (
            Path(str(state_root_value))
            if state_root_value
            else batch_root / "ray_state"
        )
        input_list = batch_root / "input_paths.txt"
        input_list.parent.mkdir(parents=True, exist_ok=True)
        input_list.write_text(
            "".join(f"{item['local_video_path']}\n" for item in selected_items),
            encoding="utf-8",
        )
        command = build_ray_command(
            params=params,
            input_list=input_list,
            output_root=normal_output_root,
            state_root=state_root,
            workers=workers,
        )
        print("Running NormalCrafter: " + " ".join(command), flush=True)
        completed = subprocess.run(
            command,
            cwd=str(REPOSITORY_ROOT),
            check=False,
        )

        output_prefix = ensure_trailing_slash(
            str(
                params.get("output_prefix")
                or f"data_output/normalcrafter_v1/{safe_item_stem(run_id)}/"
            )
        )
        flip_y = parse_bool(params.get("flip_y"), default=False)
        aggregate = aggregate_outputs(
            items=selected_items,
            output_root=normal_output_root,
            state_root=state_root,
            output_prefix=output_prefix,
            flip_y=flip_y,
        )
        results_path = batch_root / "results.jsonl"
        failed_path = batch_root / "failed_items.jsonl"
        manifest_path = batch_root / "_manifest.json"
        write_jsonl(results_path, aggregate["rows"])
        write_jsonl(failed_path, aggregate["failed_rows"])

        distinct_gpu_count = len(aggregate["used_ray_gpu_ids"])
        coverage_ok = distinct_gpu_count >= min_gpu_count
        manifest = {
            "schema_version": "normalcrafter_manifest.v1",
            "run_id": run_id,
            "source_channel": "GT",
            "predicted_channel": "predicted_normal",
            "normal_standard": "UDN-v2",
            "flip_y": flip_y,
            "output_mode": "normal-video-only",
            "prepared_items_path": str(prepared_items_path),
            "selected_item_count": len(selected_items),
            "success_count": aggregate["success_count"],
            "failure_count": aggregate["failure_count"],
            "requested_ray_worker_count": workers,
            "minimum_ray_gpu_count": min_gpu_count,
            "used_ray_gpu_ids": aggregate["used_ray_gpu_ids"],
            "used_cuda_visible_devices": aggregate["used_cuda_visible_devices"],
            "multi_gpu_coverage_ok": coverage_ok,
            "ray_batch_returncode": completed.returncode,
            "output_prefix": output_prefix,
        }
        write_json(manifest_path, manifest)

        for local_path, output_key in aggregate["success_files"]:
            output_files.append(
                {"local_path": str(local_path), "output_key": output_key}
            )
        results_key = f"{output_prefix}results.jsonl"
        failed_key = f"{output_prefix}failed_items.jsonl"
        manifest_key = f"{output_prefix}_manifest.json"
        output_files.extend(
            [
                {"local_path": str(results_path), "output_key": results_key},
                {"local_path": str(failed_path), "output_key": failed_key},
                {"local_path": str(manifest_path), "output_key": manifest_key},
            ]
        )
        artifacts = {
            **manifest,
            "manifest_key": manifest_key,
            "results_key": results_key,
            "failed_items_key": failed_key,
            "sample_item_ids": [
                str(row.get("item_id")) for row in aggregate["rows"][:5]
            ],
        }

        failure_reasons: list[str] = []
        if completed.returncode != 0:
            failure_reasons.append(
                f"ray_batch.py exited with code {completed.returncode}"
            )
        if aggregate["failure_count"]:
            failure_reasons.append(
                f"{aggregate['failure_count']} item(s) did not produce valid output"
            )
        if not coverage_ok:
            failure_reasons.append(
                f"Ray used {distinct_gpu_count} distinct GPU resource(s), "
                f"expected at least {min_gpu_count}"
            )
        status = "failed" if failure_reasons else "done"
        write_runtime_output(
            runtime_output_path,
            status=status,
            artifacts=artifacts,
            output_files=output_files,
            reason="; ".join(failure_reasons) or None,
        )
    except Exception as exc:
        write_runtime_output(
            runtime_output_path,
            status="failed",
            output_files=output_files,
            reason=f"{type(exc).__name__}: {exc}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
