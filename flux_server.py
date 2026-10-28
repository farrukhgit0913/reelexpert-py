import base64
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

HOST = "127.0.0.1"
PORT = 8765
MODEL = "black-forest-labs/FLUX.2-klein-4B"

STYLE_PROMPTS = {
    "anime": "anime illustration, expressive features, clean outlines, vivid colors",
    "ghibli": "hand-painted animated storybook illustration, soft painterly colors, lush background",
    "3d": "high-quality stylized 3D children's cartoon, rounded features, cinematic lighting, smooth shading",
    "comic": "polished comic-book illustration, bold ink outlines, expressive colors, detailed shading",
}

print("Loading FLUX.2 Klein. This can take a while on first startup.", flush=True)

pipe = Flux2KleinPipeline.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

generation_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, data):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"success": True, "model": MODEL, "ready": True})
        else:
            self.send_json(404, {"success": False, "message": "Not found"})

    def do_POST(self):
        if self.path != "/transform":
            self.send_json(404, {"success": False, "message": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 18 * 1024 * 1024:
                self.send_json(413, {"success": False, "message": "Invalid request size"})
                return

            payload = json.loads(self.rfile.read(length))
            encoded = payload.get("image")
            prompt = str(payload.get("prompt", "")).strip()[:1500]
            style = payload.get("style", "3d")

            if style not in STYLE_PROMPTS:
                self.send_json(400, {"success": False, "message": "Unsupported style"})
                return
            if not encoded:
                self.send_json(400, {"success": False, "message": "Image is required"})
                return

            raw = base64.b64decode(encoded, validate=True)
            if len(raw) > 10 * 1024 * 1024:
                self.send_json(413, {"success": False, "message": "Image exceeds 10 MB"})
                return

            with Image.open(io.BytesIO(raw)) as source:
                image = source.convert("RGB")
                image.thumbnail((1024, 1024))

            style_prompt = STYLE_PROMPTS[style]
            full_prompt = (
                f"Transform the provided input image into {style_prompt}. "
                "Preserve the main subject's identity, pose, composition, clothing, "
                "and recognizable details. Produce a coherent, polished image. "
                "Do not add text or watermarks. "
                + (f"Additional instructions: {prompt}" if prompt else "")
            )

            print(f"Generating {style} image ({image.width}x{image.height})...", flush=True)

            with generation_lock, torch.inference_mode():
                result = pipe(
                    prompt=full_prompt,
                    image=image,
                    height=max(64, (image.height // 16) * 16),
                    width=max(64, (image.width // 16) * 16),
                    num_inference_steps=4,
                    guidance_scale=1.0,
                ).images[0]

            output = io.BytesIO()
            result.save(output, format="PNG")
            png = output.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)
            print("Image generated successfully.", flush=True)

        except Exception as exc:
            print(f"Generation failed: {exc}", flush=True)
            try:
                self.send_json(500, {"success": False, "message": str(exc)[:500]})
            except (BrokenPipeError, ConnectionError):
                pass

    def log_message(self, fmt, *args):
        print(f"[HTTP] {fmt % args}", flush=True)


print(f"FLUX service listening on http://{HOST}:{PORT}", flush=True)
ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
