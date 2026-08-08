import gc
import os
import numpy as np
import torch

from diffusers.training_utils import set_seed
from diffusers import AutoencoderKLTemporalDecoder
from fire import Fire

from normalcrafter.normal_crafter_ppl import NormalCrafterPipeline
from normalcrafter.unet import DiffusersUNetSpatioTemporalConditionModelNormalCrafter
from normalcrafter.utils import (
    atomic_save_json,
    convert_normalcrafter_to_udn,
    read_video_frames,
    save_video,
    save_udn_npz,
    vis_sequence_normal,
)


class DepthCrafterDemo:
    def __init__(
        self,
        unet_path: str,
        pre_train_path: str,
        cpu_offload: str = "model",
    ):
        self.unet_path = unet_path
        self.pre_train_path = pre_train_path
        unet = DiffusersUNetSpatioTemporalConditionModelNormalCrafter.from_pretrained(
            unet_path,
            subfolder="unet",
            low_cpu_mem_usage=True,
        )
        vae = AutoencoderKLTemporalDecoder.from_pretrained(unet_path, subfolder="vae")
        weight_dtype = torch.float16
        vae.to(dtype=weight_dtype)
        unet.to(dtype=weight_dtype)
        # load weights of other components from the provided checkpoint
        self.pipe = NormalCrafterPipeline.from_pretrained(
            pre_train_path,
            unet=unet,
            vae=vae,
            torch_dtype=weight_dtype,
            variant="fp16",
        )

        # for saving memory, we can offload the model to CPU, or even run the model sequentially to save more memory
        if cpu_offload is not None:
            if cpu_offload == "sequential":
                # This will slow, but save more memory
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        else:
            self.pipe.to("cuda")
        # enable attention slicing and xformers memory efficient attention
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(e)
            print("Xformers is not enabled")
        # self.pipe.enable_attention_slicing()

    def infer(
        self,
        video: str,
        save_folder: str = "./demo_output",
        window_size: int = 14,
        time_step_size: int = 10,
        process_length: int = 195,
        decode_chunk_size: int = 7,
        max_res: int = 1024,
        dataset: str = "open",
        target_fps: int = 15,
        seed: int = 42,
        save_npz: bool = False,
        normal_standard: str = "normalcrafter",
        flip_y: bool = False,
        save_input_video: bool = True,
        save_manifest: bool = True,
    ):
        set_seed(seed)

        frames, target_fps, video_metadata = read_video_frames(
            video,
            process_length,
            target_fps,
            max_res,
            return_metadata=True,
        )
        # inference the depth map using the DepthCrafter pipeline
        with torch.inference_mode():
            res = self.pipe(
                frames,
                decode_chunk_size=decode_chunk_size,
                time_step_size=time_step_size,
                window_size=window_size,
            ).frames[0]

        is_udn = normal_standard in {"udn-v1", "udn-v2"}
        if is_udn:
            res, valid_mask = convert_normalcrafter_to_udn(res, flip_y=flip_y)
        elif normal_standard == "normalcrafter":
            res = np.asarray(res, dtype=np.float32)
            if flip_y:
                res = res.copy()
                res[..., 1] *= -1.0
            valid_mask = np.isfinite(res).all(axis=-1)
        else:
            raise ValueError(f"Unknown normal standard: {normal_standard}")

        # Visualize the normal data and save the results.
        vis = vis_sequence_normal(res)
        save_path = os.path.join(
            save_folder, os.path.splitext(os.path.basename(video))[0]
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        input_path = save_path + "_input.mp4"
        vis_path = save_path + "_vis.mp4"
        artifacts = []
        if save_input_video:
            save_video(frames, input_path, fps=target_fps)
            artifacts.append(input_path)
        save_video(
            vis,
            vis_path,
            fps=target_fps,
            data_profile=is_udn,
        )
        artifacts.append(vis_path)

        if save_npz:
            npz_path = save_path + ".npz"
            if is_udn:
                save_udn_npz(
                    npz_path,
                    normal=res,
                    valid_mask=valid_mask,
                    standard=np.array("UDN-v2"),
                    conformance=np.array(not flip_y),
                    flip_y=np.array(flip_y),
                    coordinate_transform=np.array(
                        "diag(1,-1,1)" if flip_y else "identity"
                    ),
                    signed_z=np.array(True),
                    hemisphere_canonicalization=np.array("none"),
                    coordinate_system=np.array(
                        (
                            "legacy-x-right-y-reflected-z-toward-camera"
                            if flip_y
                            else "right-handed-camera-x-right-y-up-z-toward-camera"
                        )
                    ),
                )
            else:
                np.savez_compressed(npz_path, depth=res)
            artifacts.append(npz_path)

        if is_udn and save_manifest:
            manifest_path = save_path + "_manifest.json"
            manifest = {
                "standard": "UDN-v2",
                "schema_version": 3,
                "conformance": not flip_y,
                "compliance": "Canonical" if not flip_y else "Legacy",
                "depth_source": None,
                "normal_source": {
                    "type": "NormalCrafter RGB inference",
                    "unet": self.unet_path,
                    "pretrained_pipeline": self.pre_train_path,
                },
                "normal_conversion": {
                    "source_coordinates": "right-handed-camera-x-right-y-up-z-toward-camera",
                    "target_coordinates": (
                        "legacy-x-right-y-reflected-z-toward-camera"
                        if flip_y
                        else "right-handed-camera-x-right-y-up-z-toward-camera"
                    ),
                    "coordinate_transform": ("diag(1,-1,1)" if flip_y else "identity"),
                    "flip_y": flip_y,
                    "operations": (["reflect-y"] if flip_y else [])
                    + ["normalize-float32", "invalid-to-0-0-1"],
                    "signed_z_preserved": True,
                    "hemisphere_canonicalization": "none",
                },
                "validity_policy": (
                    "finite non-zero model predictions are valid; no independent "
                    "background/depth mask is available"
                ),
                "video": {
                    **video_metadata,
                    "frame_count": int(res.shape[0]),
                    "width": int(res.shape[2]),
                    "height": int(res.shape[1]),
                    "crop_history": [],
                    "resize_history": (
                        []
                        if video_metadata["source_width"]
                        == video_metadata["output_width"]
                        and video_metadata["source_height"]
                        == video_metadata["output_height"]
                        else [
                            {
                                "algorithm": "decord resize",
                                "from": [
                                    video_metadata["source_width"],
                                    video_metadata["source_height"],
                                ],
                                "to": [
                                    video_metadata["output_width"],
                                    video_metadata["output_height"],
                                ],
                            }
                        ]
                    ),
                },
                "artifacts": {
                    "canonical_normal": (
                        os.path.basename(save_path + ".npz") if save_npz else None
                    ),
                    "normal_preview": os.path.basename(vis_path),
                    "input_preview": (
                        os.path.basename(input_path) if save_input_video else None
                    ),
                },
                "normal_data": {
                    "dtype": "float32",
                    "shape": list(res.shape),
                    "range": [-1.0, 1.0],
                    "unit_length": True,
                    "conformance": not flip_y,
                    "flip_y": flip_y,
                    "signed_z": True,
                    "hemisphere_canonicalization": "none",
                    "npz_key": "normal" if save_npz else None,
                    "valid_mask_key": "valid_mask" if save_npz else None,
                },
                "preview_transport": {
                    "compliance": "Transport-approximate",
                    "coordinate_transform": ("diag(1,-1,1)" if flip_y else "identity"),
                    "semantic_rgb": "normal*0.5+0.5 without EOTF/OETF",
                    "codec": "H.264",
                    "pixel_format": "yuv420p",
                    "bit_depth": 8,
                    "color_matrix": "BT.709",
                    "color_range": "Limited",
                    "decode": "normalize(2*RGB-1)",
                    "signed_z": True,
                },
            }
            atomic_save_json(manifest_path, manifest)
            artifacts.append(manifest_path)

        return artifacts

    def run(
        self,
        input_video,
        num_denoising_steps,
        guidance_scale,
        max_res=1024,
        process_length=195,
    ):
        res_path = self.infer(
            input_video,
            num_denoising_steps,
            guidance_scale,
            max_res=max_res,
            process_length=process_length,
        )
        # clear the cache for the next video
        gc.collect()
        torch.cuda.empty_cache()
        return res_path[:2]


def main(
    video_path: str,
    save_folder: str = "./demo_output",
    unet_path: str = "Yanrui95/NormalCrafter",
    pre_train_path: str = "stabilityai/stable-video-diffusion-img2vid-xt",
    process_length: int = -1,
    cpu_offload: str = "model",
    target_fps: int = -1,
    seed: int = 42,
    window_size: int = 14,
    time_step_size: int = 10,
    max_res: int = 1024,
    dataset: str = "open",
    save_npz: bool = False,
    normal_standard: str = "normalcrafter",
    flip_y: bool = False,
    save_input_video: bool = True,
    save_manifest: bool = True,
):
    depthcrafter_demo = DepthCrafterDemo(
        unet_path=unet_path,
        pre_train_path=pre_train_path,
        cpu_offload=cpu_offload,
    )
    # process the videos, the video paths are separated by comma
    video_paths = video_path.split(",")
    for video in video_paths:
        depthcrafter_demo.infer(
            video,
            save_folder=save_folder,
            window_size=window_size,
            process_length=process_length,
            time_step_size=time_step_size,
            max_res=max_res,
            dataset=dataset,
            target_fps=target_fps,
            seed=seed,
            save_npz=save_npz,
            normal_standard=normal_standard,
            flip_y=flip_y,
            save_input_video=save_input_video,
            save_manifest=save_manifest,
        )
        # clear the cache for the next video
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # running configs
    # the most important arguments for memory saving are `cpu_offload`, `enable_xformers`, `max_res`, and `window_size`
    # the most important arguments for trade-off between quality and speed are
    # `num_inference_steps`, `guidance_scale`, and `max_res`
    Fire(main)
