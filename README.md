# NERVE — local app

**NERVE links:** [models & configs](https://huggingface.co/Phips/NERVE) · [try it online (ZeroGPU)](https://huggingface.co/spaces/Phips/nerve) · [convert a checkpoint](https://huggingface.co/spaces/Phips/nerve-convert) · [local app](https://github.com/Phhofm/nerve-app) · [train your own (Colab)](https://github.com/Phhofm/nerve-train)

A local GUI for the released [NERVE](https://huggingface.co/Phips/NERVE) models. Upscale a
single image, or a whole folder, and save uncompressed PNGs. Drop an image, pick a
model, click **Upscale** — models download automatically on first use.

## Easiest: download and double-click

Grab the ready-made build for your OS from the
**[Releases page](https://github.com/Phhofm/nerve-app/releases)**:

| OS | Download |
|---|---|
| Windows | `NERVE-Windows.zip` |
| Linux | `NERVE-Linux.zip` |
| macOS | `NERVE-macOS.zip` |

Unzip it and run **`NERVE`** — a browser tab opens with the app. No Python, no
terminal, no CUDA install.

> Don't want to install anything at all? Use the **online demo**:
> [huggingface.co/spaces/Phips/nerve](https://huggingface.co/spaces/Phips/nerve)

## GPU support (in the packaged build)

| OS | Accelerator | Setup needed |
|---|---|---|
| Windows | **DirectML** — any DX12 GPU (NVIDIA / AMD / Intel) | none |
| macOS | **CoreML** (Apple Silicon GPU / Neural Engine) | none |
| Linux | **CPU** by default; NVIDIA if you install `onnxruntime-gpu` | optional |
| any | CPU fallback | none |

The app shows a **status banner** at the top:
- **green** — running on the GPU (CUDA / DirectML / CoreML / OpenVINO);
- **amber** — a GPU was detected but the GPU build of ONNX Runtime isn't installed, so it's on CPU (it tells you the exact command to fix it);
- **grey** — no GPU detected, running on CPU.

## If you already have Python

One-click launcher: run **`start.bat`** (Windows) or **`./start.sh`** (Linux/macOS).
It creates a venv, installs the dependencies and launches the app.

Or manually:

```bash
pip install -r requirements.txt
# optional: pip install onnxruntime-gpu      # NVIDIA
# optional: pip install onnxruntime-openvino # Intel
python app.py
```

## What it does

- **Single image** tab — drag an image in, press **Upscale**, download the PNG.
- **Folder (batch)** tab — point at an input folder and an output folder; every
  image is upscaled and written as an uncompressed PNG.

- **2x / 4x release** — for clean images.
- **2x / 4x OTF GAN** — for degraded images (blurry, noisy, JPEG-compressed).

Apache-2.0.
