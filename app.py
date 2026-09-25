"""NERVE — local upscaler (single image or a whole folder).

Uses ONNX Runtime, so it runs on whatever accelerator you have:
NVIDIA (CUDA), Windows GPUs incl. Intel/AMD (DirectML), Apple Silicon (CoreML),
Intel (OpenVINO), or plain CPU. Pick the fastest available automatically.

Run:  python app.py
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "Phips/NERVE"
MODELS: dict[str, tuple[str, int]] = {
    "NERVE 2x — release (clean input)": ("onnx/2x_NERVE_1x3xHxW_fp32_op20.onnx", 2),
    "NERVE 4x — release (clean input)": ("onnx/4x_NERVE_1x3xHxW_fp32_op20.onnx", 4),
    "NERVE 4x — OTF GAN (blurry / noisy / JPEG input)": (
        "onnx/4x_NERVE_OTF_gan_1x3xHxW_fp32_op20.onnx",
        4,
    ),
}

# Fastest first. Only providers actually available on this machine are used.
PROVIDERS = [
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "OpenVINOExecutionProvider",
    "CPUExecutionProvider",
]

_sessions: dict[str, ort.InferenceSession] = {}
TILE = 512
OVERLAP = 32

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def get_session(key: str) -> ort.InferenceSession:
    if key not in _sessions:
        path, _ = MODELS[key]
        local = hf_hub_download(REPO, path)
        available = ort.get_available_providers()
        providers = [p for p in PROVIDERS if p in available] or ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _sessions[key] = ort.InferenceSession(local, so, providers=providers)
    return _sessions[key]


def active_provider() -> str:
    if not _sessions:
        return "—"
    return _sessions[next(iter(_sessions))].get_providers()[0]


def upscale_array(session: ort.InferenceSession, scale: int, rgb: np.ndarray) -> np.ndarray:
    """rgb: float32 HWC in [0,1] -> float32 HWC in [0,1], using tiling."""
    h, w = rgb.shape[:2]
    out = np.zeros((h * scale, w * scale, 3), dtype=np.float32)
    in_name = session.get_inputs()[0].name
    step = TILE - OVERLAP
    for y in range(0, h, step):
        for x in range(0, w, step):
            y1, x1 = min(y + TILE, h), min(x + TILE, w)
            y0, x0 = max(0, y1 - TILE), max(0, x1 - TILE)
            crop = rgb[y0:y1, x0:x1].transpose(2, 0, 1)[None].astype(np.float32)
            y_hat = session.run(None, {in_name: crop})[0][0].transpose(1, 2, 0)
            out[y0 * scale : y1 * scale, x0 * scale : x1 * scale] = np.clip(y_hat, 0, 1)
    return out


def _to_png(arr: np.ndarray, path: str) -> None:
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(path)


def upscale_single(image: Image.Image, model_key: str) -> tuple[Image.Image, str]:
    if image is None:
        raise gr.Error("Please upload an image first.")
    session = get_session(model_key)
    scale = MODELS[model_key][1]
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    out = upscale_array(session, scale, rgb)
    png = "/tmp/nerve_output.png"
    _to_png(out, png)
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), mode="RGB"), png


def upscale_folder(in_dir: str, out_dir: str, model_key: str, progress=gr.Progress()):
    if not in_dir or not out_dir:
        raise gr.Error("Set both an input and an output folder.")
    src, dst = Path(in_dir), Path(out_dir)
    if not src.is_dir():
        raise gr.Error(f"Input folder not found: {in_dir}")
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_TYPES)
    if not files:
        raise gr.Error(f"No images found in {in_dir}")
    session = get_session(model_key)
    scale = MODELS[model_key][1]
    for i, f in enumerate(files):
        progress((i + 1) / len(files), desc=f"{f.name}")
        rgb = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
        _to_png(upscale_array(session, scale, rgb), str(dst / f"{f.stem}.png"))
    return f"Done: {len(files)} image(s) -> {out_dir}"


with gr.Blocks(title="NERVE") as demo:
    gr.Markdown(
        """
        # NERVE

        Lightweight super-resolution / restoration. Everything runs locally on
        ONNX Runtime — NVIDIA (CUDA), Windows GPUs incl. Intel/AMD (DirectML),
        Apple Silicon (CoreML), Intel (OpenVINO), or CPU.

        **What is NERVE?** A small (~1.8M params) pure-convolution network built for
        *ease of use*, not benchmark chasing: one file, one variant, and a checkpoint
        that converts cleanly to dynamic ONNX, TensorRT and ncnn — no fused/unfused
        pairs. Models and details: **[huggingface.co/Phips/NERVE](https://huggingface.co/Phips/NERVE)**.

        - **release** models → **clean** images.
        - the **OTF GAN** model → **degraded** images (blurry, noisy, JPEG-compressed).
        """
    )
    model = gr.Dropdown(choices=list(MODELS), value=list(MODELS)[1], label="Model")
    provider = gr.Textbox(label="Accelerator", interactive=False)

    with gr.Tab("Single image"):
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Input image", height=320)
                btn = gr.Button("Upscale", variant="primary")
            with gr.Column():
                out = gr.Image(type="pil", label="Output (uncompressed PNG)", height=320)
                dl = gr.File(label="Download PNG")

        def _single(image, key):
            img, path = upscale_single(image, key)
            return img, path, active_provider()

        btn.click(_single, inputs=[inp, model], outputs=[out, dl, provider])

    with gr.Tab("Folder (batch)"):
        gr.Markdown("Upscale every image in a folder and save uncompressed PNGs.")
        with gr.Row():
            in_dir = gr.Textbox(label="Input folder", placeholder="/path/to/images")
            out_dir = gr.Textbox(label="Output folder", placeholder="/path/to/output")
        run = gr.Button("Upscale folder", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)

        def _folder(i, o, key):
            return upscale_folder(i, o, key), active_provider()

        run.click(_folder, inputs=[in_dir, out_dir, model], outputs=[status, provider])


if __name__ == "__main__":
    # inbrowser=True -> double-clicking the packaged app opens the UI in a browser tab
    demo.queue().launch(server_name="127.0.0.1", inbrowser=True)
