import base64
import binascii
import io
import json
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
    "clay": "handcrafted clay animation, sculpted clay textures, miniature set, warm studio lighting",
    "realistic": "photorealistic image, natural details, realistic textures, balanced professional lighting",
    "fantasy": "beautiful fantasy concept illustration, imaginative details, atmospheric cinematic lighting",
    "product": "premium commercial product photography, clean studio lighting, refined composition, soft shadows",
}

MODE_INSTRUCTIONS = {
    "photo-cartoon": "Transform the provided input image into the requested visual style. Preserve the subject's recognizable identity, pose, clothing and composition as closely as possible.",
    "sketch-image": "Turn the provided sketch or drawing into a polished, fully rendered illustration. Respect the original drawing's main shapes and composition.",
    "background-restyle": "Restyle the environment and background of the provided image according to the request. Keep the main subject, pose, clothing and recognizable details as close to the input as possible. This is a best-effort edit, not a pixel-accurate mask-based cutout.",
    "character-designer": "Design one original, appealing character from the description. Show a clear, coherent character design with readable clothing, silhouette and facial features.",
    "thumbnail": "Create a polished, high-impact social video thumbnail image with a clear focal point, strong composition and readable visual hierarchy. Avoid rendering words or lettering; leave clean space where a title can be added later.",
    "story-scenes": "Create one polished, family-friendly story scene from the description. Use expressive characters, clear staging and a visually readable composition.",
    "product-studio": "Restyle the supplied product image as premium commercial product photography. Preserve the product's shape and important visible details as closely as possible. Do not invent readable labels or logos.",
    "character-variations": "Create a variation of the character shown in the input image using the requested changes. Preserve the character's recognizable face, hairstyle, proportions and signature design details as closely as possible.",
}

ASPECTS = {
    "portrait": (768, 1024),
    "vertical": (576, 1024),
    "square": (768, 768),
    "landscape": (1024, 768),
    "widescreen": (1024, 576),
}

print("Loading FLUX.2 Klein. This can take a while on first startup.", flush=True)
pipe = Flux2KleinPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
generation_lock = threading.Lock()


def json_bytes(data):
    return json.dumps(data).encode("utf-8")


def rounded_dimension(value):
    return max(64, (int(value) // 16) * 16)


def generate_image(payload, legacy=False):
    encoded = payload.get("image")
    mode = "photo-cartoon" if legacy else str(payload.get("mode", ""))
    style = str(payload.get("style", "3d"))
    prompt = str(payload.get("prompt", "")).strip()[:1500]
    aspect = str(payload.get("aspectRatio", "portrait"))

    if style not in STYLE_PROMPTS:
        raise ValueError("Unsupported visual style.")
    if mode not in MODE_INSTRUCTIONS:
        raise ValueError("Unsupported Image Studio feature.")
    if aspect not in ASPECTS:
        raise ValueError("Unsupported aspect ratio.")

    image = None
    if encoded:
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("The uploaded image is not valid base64 data.") from exc
        if len(raw) > 10 * 1024 * 1024:
            raise ValueError("Image exceeds 10 MB.")
        try:
            with Image.open(io.BytesIO(raw)) as source:
                image = source.convert("RGB")
        except Exception as exc:
            raise ValueError("Could not decode the uploaded image.") from exc

    image_required = mode in {"photo-cartoon", "sketch-image", "background-restyle", "product-studio", "character-variations"}
    if image_required and image is None:
        raise ValueError("This feature requires a source image.")
    if image is None and not prompt:
        raise ValueError("Provide a description or upload a reference image.")

    if image is not None:
        # Preserve the uploaded image's aspect ratio and cap the longest edge to limit memory use.
        image.thumbnail((768, 1024), Image.Resampling.LANCZOS)
        width = rounded_dimension(image.width)
        height = rounded_dimension(image.height)
    else:
        width, height = ASPECTS[aspect]

    full_prompt = (
        f"{MODE_INSTRUCTIONS[mode]} Visual direction: {STYLE_PROMPTS[style]}. "
        "Create a coherent, polished image. Avoid text, letters, subtitles, logos and watermarks. "
        + (f"User instructions: {prompt}" if prompt else "")
    )[:3000]

    print(f"Generating {mode} / {style} at {width}x{height} (image input: {image is not None})...", flush=True)
    with generation_lock, torch.inference_mode():
        kwargs = {
            "prompt": full_prompt,
            "height": height,
            "width": width,
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
        }
        if image is not None:
            kwargs["image"] = image
        result = pipe(**kwargs).images[0]

    output = io.BytesIO()
    result.save(output, format="PNG")
    return output.getvalue()


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, data):
        payload = json_bytes(data)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"success": True, "model": MODEL, "ready": True, "features": list(MODE_INSTRUCTIONS)})
        else:
            self.send_json(404, {"success": False, "message": "Not found"})

    def do_POST(self):
        legacy = self.path == "/transform"
        if not legacy and self.path != "/generate":
            self.send_json(404, {"success": False, "message": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 18 * 1024 * 1024:
                self.send_json(413, {"success": False, "message": "Invalid request size"})
                return
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                self.send_json(400, {"success": False, "message": "Expected a JSON object."})
                return
            if legacy and not payload.get("image"):
                self.send_json(400, {"success": False, "message": "Image is required."})
                return
            png = generate_image(payload, legacy=legacy)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(png)
            print("Image generated successfully.", flush=True)
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json(400, {"success": False, "message": str(exc)[:500]})
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
