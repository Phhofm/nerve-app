"""NERVE — local upscaler.

Run:  python app.py   (then open the printed local URL)

Pick a model, drop an image, press Upscale, download the PNG. Uses your GPU
if you have one (CUDA), otherwise falls back to CPU.
"""

from __future__ import annotations

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file

from nerve_arch import nerve

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REPO = "Phips/NERVE"
MODELS: dict[str, tuple[str, int]] = {
    "NERVE 2x — release (clean input)": ("models/2x_NERVE_release.safetensors", 2),
    "NERVE 4x — release (clean input)": ("models/4x_NERVE_release.safetensors", 4),
    "NERVE 4x — OTF GAN (blurry / noisy / JPEG input)": (
        "models/4x_NERVE_OTF_gan.safetensors",
        4,
    ),
}

_cache: dict[str, torch.nn.Module] = {}


def get_model(key: str) -> torch.nn.Module:
    if key not in _cache:
        path, scale = MODELS[key]
        model = nerve(scale=scale)
        model.load_state_dict(
            load_file(hf_hub_download(REPO, path), device="cpu"), strict=True
        )
        _cache[key] = model.eval()
    return _cache[key]


@torch.inference_mode()
def _run(model: torch.nn.Module, img: np.ndarray, scale: int) -> np.ndarray:
    """img: float32 HWC RGB in [0,1]; returns float32 HWC RGB in [0,1]."""
    h, w = img.shape[:2]
    tile = 512  # LR tile size; keeps VRAM low on ZeroGPU
    overlap = 32
    out = np.zeros((h * scale, w * scale, 3), dtype=np.float32)
    for y in range(0, h, tile - overlap):
        for x in range(0, w, tile - overlap):
            y1, x1 = min(y + tile, h), min(x + tile, w)
            y0, x0 = max(0, y1 - tile), max(0, x1 - tile)
            crop = img[y0:y1, x0:x1]
            t = torch.from_numpy(crop.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            y_hat = model(t).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
            out[y0 * scale : y1 * scale, x0 * scale : x1 * scale] = y_hat
    return out


def upscale(image: Image.Image, model_key: str) -> Image.Image:
    if image is None:
        raise gr.Error("Please upload an image first.")
    model = get_model(model_key).to(DEVICE)
    scale = MODELS[model_key][1]

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    result = _run(model, rgb, scale)
    return Image.fromarray((result * 255.0 + 0.5).astype(np.uint8), mode="RGB")


with gr.Blocks(title="NERVE", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # NERVE
        Lightweight super-resolution / restoration. Drop an image, pick a model,
        press **Upscale**, download the full-quality PNG.

        - **release** models are for **clean** images.
        - the **OTF GAN** model is for **degraded** images (blurry, noisy, JPEG-y).
        """
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Input image", height=320)
            model = gr.Dropdown(
                choices=list(MODELS),
                value=list(MODELS)[1],
                label="Model",
            )
            btn = gr.Button("Upscale", variant="primary")
        with gr.Column():
            out = gr.Image(type="pil", label="Output (uncompressed PNG)", height=320)
            dl = gr.File(label="Download PNG")

    def _wrap(image, model_key):  # noqa: ANN202
        result = upscale(image, model_key)
        result.save("/tmp/nerve_output.png")
        return result, "/tmp/nerve_output.png"

    btn.click(_wrap, inputs=[inp, model], outputs=[out, dl])

if __name__ == "__main__":
    demo.queue().launch()
