# NERVE — local app

A tiny local GUI for the released [NERVE](https://huggingface.co/Phips/NERVE) models.

## Run

```bash
pip install -r requirements.txt
python app.py
```

Then open the local URL Gradio prints. Pick a model, drop in an image, press
**Upscale**, and download the uncompressed PNG.

- **2x / 4x release** — for clean images.
- **4x OTF GAN** — for degraded images (blurry, noisy, JPEG-compressed).

Models are downloaded from `Phips/NERVE` on first use and cached. Uses CUDA if
available, otherwise CPU. Large images are processed in tiles.

Apache-2.0.
