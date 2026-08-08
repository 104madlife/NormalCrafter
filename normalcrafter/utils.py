import json
import os
import tempfile
import zipfile
from typing import List, Union

import numpy as np
import PIL.Image
import mediapy
from decord import VideoReader, cpu


def read_video_frames(
    video_path, process_length, target_fps, max_res, return_metadata=False
):
    print("==> processing video: ", video_path)
    vid = VideoReader(video_path, ctx=cpu(0))
    print("==> original video shape: ", (len(vid), *vid.get_batch([0]).shape[1:]))
    source_frame_count = len(vid)
    source_fps = float(vid.get_avg_fps())
    original_height, original_width = vid.get_batch([0]).shape[1:3]

    if max(original_height, original_width) > max_res:
        scale = max_res / max(original_height, original_width)
        height = round(original_height * scale)
        width = round(original_width * scale)
    else:
        height = original_height
        width = original_width

    vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

    fps = source_fps if target_fps == -1 else target_fps
    stride = round(source_fps / fps)
    stride = max(stride, 1)
    frames_idx = list(range(0, len(vid), stride))
    print(
        f"==> downsampled shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}, with stride: {stride}"
    )
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
    print(
        f"==> final processing shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}"
    )
    frames = vid.get_batch(frames_idx).asnumpy().astype(np.uint8)
    frames = [PIL.Image.fromarray(x) for x in frames]

    if return_metadata:
        metadata = {
            "source_width": int(original_width),
            "source_height": int(original_height),
            "source_frame_count": int(source_frame_count),
            "source_fps": source_fps,
            "output_width": int(width),
            "output_height": int(height),
            "output_fps": float(fps),
            "source_frame_indices": frames_idx,
            "source_timestamps_seconds": [index / source_fps for index in frames_idx],
            "sampling_stride": int(stride),
            "max_res": int(max_res),
        }
        return frames, fps, metadata
    return frames, fps


def normalize_normal_vectors(normals: np.ndarray, eps: float = 1e-6):
    """Normalize signed normals and replace invalid vectors with the UDN sentinel."""
    source = np.asarray(normals, dtype=np.float32)
    if source.ndim < 1 or source.shape[-1] != 3:
        raise ValueError(f"Expected normals with last dimension 3, got {source.shape}")

    finite = np.isfinite(source).all(axis=-1)
    safe = np.where(finite[..., None], source, 0.0)
    lengths = np.linalg.norm(safe, axis=-1, keepdims=True)
    valid = finite & (lengths[..., 0] > eps)
    safe /= np.maximum(lengths, eps)
    safe[~valid] = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    return np.ascontiguousarray(safe, dtype=np.float32), valid


def convert_normalcrafter_to_udn(
    normals: np.ndarray, eps: float = 1e-6, flip_y: bool = False
):
    """Normalize NormalCrafter normals, optionally applying a fixed Y reflection."""
    converted = np.asarray(normals, dtype=np.float32).copy()
    if converted.ndim < 1 or converted.shape[-1] != 3:
        raise ValueError(
            f"Expected normals with last dimension 3, got {converted.shape}"
        )
    if flip_y:
        converted[..., 1] *= -1.0
    return normalize_normal_vectors(converted, eps=eps)


def decode_udn_rgb(rgb: np.ndarray, eps: float = 1e-6):
    """Decode independent UDN RGB data without hemisphere canonicalization."""
    encoded = np.asarray(rgb)
    if encoded.ndim < 1 or encoded.shape[-1] != 3:
        raise ValueError(f"Expected RGB with last dimension 3, got {encoded.shape}")
    if np.issubdtype(encoded.dtype, np.unsignedinteger):
        encoded = encoded.astype(np.float32) / np.iinfo(encoded.dtype).max
    elif np.issubdtype(encoded.dtype, np.signedinteger):
        raise ValueError("Signed integer RGB input is not supported")
    else:
        encoded = encoded.astype(np.float32, copy=False)
    decoded = np.clip(encoded, 0.0, 1.0) * 2.0 - 1.0
    return normalize_normal_vectors(decoded, eps=eps)


def save_udn_npz(path: str, **arrays):
    """Write a standard NPZ using fast Deflate while preserving NumPy metadata."""
    output_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=os.path.dirname(output_path),
    )
    os.close(fd)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=1,
            allowZip64=True,
        ) as archive:
            for key, value in arrays.items():
                with archive.open(f"{key}.npy", "w", force_zip64=True) as handle:
                    np.lib.format.write_array(
                        handle, np.asanyarray(value), allow_pickle=False
                    )
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_video(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]],
    output_video_path: str = None,
    fps: int = 10,
    crf: int = 18,
    data_profile: bool = False,
) -> str:
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    if isinstance(video_frames[0], np.ndarray):
        video_frames = [
            np.rint(np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
            for frame in video_frames
        ]

    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    writer_options = {}
    if data_profile:
        writer_options = {
            "encoded_format": "yuv420p",
            "ffmpeg_args": [
                "-vf",
                "scale=in_range=full:out_range=limited:out_color_matrix=bt709",
                "-colorspace",
                "bt709",
                "-color_range",
                "tv",
            ],
        }
    mediapy.write_video(
        output_video_path, video_frames, fps=fps, crf=crf, **writer_options
    )
    return output_video_path


def vis_sequence_normal(normals: np.ndarray):
    normals = normals.clip(-1.0, 1.0)
    normals = normals * 0.5 + 0.5
    return normals


def atomic_save_json(path: str, payload: dict):
    output_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=os.path.dirname(output_path),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
