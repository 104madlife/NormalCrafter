## ___***NormalCrafter: Learning Temporally Consistent Video Normal from Video Diffusion Priors***___
<div align="center">
<br>

_**[Yanrui Bin<sup>1</sup>](https://binyr.github.io/),[Wenbo Hu<sup>2*](https://wbhu.github.io), 
[Haoyuan Wang<sup>3](https://www.whyy.site/), 
[Xinya Chen<sup>4](https://xinyachen21.github.io/), 
[Bing Wang<sup>2 &dagger;</sup>](https://bingcs.github.io/)**_
<br>
<sup>1</sup>Spatial Intelligence Group, The Hong Kong Polytechnic University
<sup>2</sup>ARC Lab, Tencent PCG
<sup>3</sup>City University of Hong Kong
<sup>4</sup>Huazhong University of Science and Technology
<!-- </div> -->

ICCV 2025

![Version](https://img.shields.io/badge/version-1.0.0-blue) &nbsp;
 <a href='https://arxiv.org/abs/2504.11427'><img src='https://img.shields.io/badge/arXiv-2504.01016-b31b1b.svg'></a> &nbsp;
 <a href='https://normalcrafter.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
 <a href='https://huggingface.co/spaces/Yanrui95/NormalCrafter'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> &nbsp;

</div>

## 🔆 Notice
We recommend that everyone use English to communicate on issues, as this helps developers from around the world discuss, share experiences, and answer questions together.

For business licensing and other related inquiries, don't hesitate to contact `binyanrui@gmail.com`.

## 🔆 Introduction
🤗 If you find NormalCrafter useful, **please help ⭐ this repo**, which is important to Open-Source projects. Thanks!

🔥 NormalCrafter can generate temporally consistent normal sequences
with fine-grained details from open-world videos with arbitrary lengths.

- `[24-04-01]` 🔥🔥🔥 **NormalCrafter** is released now, have fun!
## 🚀 Quick Start

### 🤖 Gradio Demo
- Online demo: [NormalCrafter](https://huggingface.co/spaces/Yanrui95/NormalCrafter) 
- Local demo:
    ```bash
    gradio app.py
    ``` 

### 🛠️ Installation
1. Clone this repo:
```bash
git clone git@github.com:Binyr/NormalCrafter.git
```
2. Install dependencies (please refer to [requirements.txt](requirements.txt)):
```bash
pip install -r requirements.txt
```



### 🤗 Model Zoo
[NormalCrafter](https://huggingface.co/Yanrui95/NormalCrafter) is available in the Hugging Face Model Hub.

### 🏃‍♂️ Inference
#### 1. High-resolution inference, requires a GPU with ~20GB memory for 1024x576 resolution:
```bash
python run.py  --video-path examples/example_01.mp4
```

#### 2. Low-resolution inference requires a GPU with ~6GB memory for 512x256 resolution:
```bash
python run.py  --video-path examples/example_01.mp4 --max-res 512
```

### Multi-GPU batch inference

Install the optional Ray dependency and process a directory with one persistent
worker per GPU:

```bash
pip install -r requirements-batch.txt
python ray_batch.py \
  --input /data/videos \
  --output /data/normalcrafter-results \
  --workers 8 \
  --retries 2
```

Completed videos are validated and recorded under `.done` in the output
directory. Running the same command again skips them without loading the model.
Exhausted failures are recorded under `.failed`; pass `--retry-failed` to retry
them. Use `--input-list paths.txt` for a newline-separated manifest and
`--dry-run` to inspect pending work without starting Ray.

To write canonical UDN-v2 camera normals, enable both the standard conversion
and float NPZ output:

```bash
python ray_batch.py \
  --input /data/videos \
  --output /data/normalcrafter-udn \
  --workers 8 \
  --save-npz \
  --normal-standard udn-v2 \
  --offline
```

Each job writes a float32 NPZ containing `normal` and `valid_mask`, a UDN-v2
manifest, and an H.264 preview. The NPZ is the canonical artifact; the preview
is transport-approximate. By default the conversion preserves NormalCrafter's
axis directions, normalizes the vectors, and preserves signed `Nz`.
Independent RGB normals must be decoded with `normalize(2*RGB-1)`; do not flip
vectors per pixel when decoded `Nz` is negative.

For a compact dataset containing only same-name UDN normal MP4 files, keep the
resume state outside the output directory:

```bash
python ray_batch.py \
  --input /data/videos \
  --output /data/normal-videos \
  --state-dir /data/normal-videos-state \
  --output-mode normal-video-only \
  --normal-standard udn-v2 \
  --workers 8 \
  --offline
```

This mode preserves the input-relative filename, writes BT.709 Limited
H.264/yuv420p, and leaves only MP4 files under `--output`. Because the transport
is lossy 8-bit YUV420, these files are UDN-v2 transport-approximate rather than
canonical float normals. Batch schema 6 records the Y transform and signed-Z
semantics, so incompatible earlier outputs are scheduled for regeneration.

The optional `--flip-y` switch applies the fixed transform `Ny=-Ny`. It is off
by default, leaving NormalCrafter ground normals green. Enabling it produces the
legacy Y-reflected convention with magenta/red ground and marks the manifest as
non-conformant. `--no-flip-y` explicitly selects the default behavior. The old
`--normal-standard udn-v1` spelling remains accepted as a deprecated alias for
`udn-v2`.
