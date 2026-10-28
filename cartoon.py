import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

MODEL = "black-forest-labs/FLUX.2-klein-4B"

if not torch.backends.mps.is_available():
    raise RuntimeError(
        "Apple Metal (MPS) is unavailable in this PyTorch environment."
    )

pipe = Flux2KleinPipeline.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
)

# Try CPU offloading if supported by your installed version.
pipe.enable_model_cpu_offload()

image = Image.open("input.jpg").convert("RGB")

prompt = (
    "Transform this photo into a beautiful hand-painted "
    "animated storybook illustration, soft colors, expressive "
    "features, lush background, cinematic lighting. "
    "Preserve the person's identity, pose and composition."
)

result = pipe(
    prompt=prompt,
    image=image,
    num_inference_steps=4,
    guidance_scale=1.0,
).images[0]

result.save("cartoon-output.png")
print("Saved cartoon-output.png")