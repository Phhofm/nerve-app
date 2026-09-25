# NERVE — local app

**NERVE links:** [models & configs](https://huggingface.co/Phips/NERVE) · [try it online (ZeroGPU)](https://huggingface.co/spaces/Phips/nerve) · [local app](https://github.com/Phhofm/nerve-app) · [train your own (Colab)](https://github.com/Phhofm/nerve-train)

A small local GUI for the released [NERVE](https://huggingface.co/Phips/NERVE) models.
Upscale a single image, or a whole folder, and save uncompressed PNGs.

## Install & run

```bash
pip install -r requirements.txt
python app.py
```

Then open the local URL Gradio prints.

## GPU support

The app uses **ONNX Runtime**, so it runs on whatever accelerator you have.
Install the one that matches your hardware (plain `onnxruntime` is CPU):

| Hardware | Install |
|---|---|
| NVIDIA (CUDA) | `pip install onnxruntime-gpu` |
| Windows: any GPU incl. Intel/AMD | `pip install onnxruntime-directml` |
| Intel (OpenVINO) | `pip install onnxruntime-openvino` |
| Apple Silicon | plain `onnxruntime` (uses CoreML) |
| CPU only | plain `onnxruntime` |

The app picks the fastest available provider automatically and shows the active
one in the UI. Works on Linux, Windows and macOS.

## What it does

- **Single image** tab — drop an image, press Upscale, download the PNG.
- **Folder (batch)** tab — point at an input folder and an output folder; every
  image is upscaled and written as an uncompressed PNG.

Models are downloaded from `Phips/NERVE` on first use and cached. Large images
are processed in tiles, so memory stays low.

- **2x / 4x release** — for clean images.
- **4x OTF GAN** — for degraded images (blurry, noisy, JPEG-compressed).

Apache-2.0.
