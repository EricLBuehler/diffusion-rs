import torch
from diffusers import FluxPipeline
import time

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("mps")

prompt = "Draw a picture of a beautiful sunset in the winter in the mountains, 4k, high quality."
start = time.time()
image = pipe(
    prompt,
    guidance_scale=3.5, #0,
    num_inference_steps=50, #4,
    height=720,
    width=1280,
    max_sequence_length=256,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
end = time.time()
print(end - start)
image.save("flux-dev.png")

# schnell: 36s (31)
# dev: 392s (389)