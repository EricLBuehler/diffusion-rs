import torch
from diffusers import FluxTransformer2DModel, FluxPipeline

model_id = "black-forest-labs/FLUX.1-dev"
nf4_id = "sayakpaul/flux.1-dev-nf4-with-bnb-integration"
model_nf4 = FluxTransformer2DModel.from_pretrained(nf4_id, torch_dtype=torch.bfloat16)
print(model_nf4.dtype)
print(model_nf4.config.quantization_config)

pipe = FluxPipeline.from_pretrained(model_id, transformer=model_nf4, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

prompt = "Draw a picture of a beautiful sunset in the winter in the mountains."
image = pipe(prompt, guidance_scale=3.5, num_inference_steps=50, generator=torch.manual_seed(0)).images[0]
image.save("flux-nf4-dev-loaded.png")
