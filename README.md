<a name="top"></a>
<h1 align="center">
  diffuse-rs
</h1>

<h3 align="center">
Blazingly fast inference of diffusion models.
</h3>

## Features
- Quantization
  - `bitsandbytes` format (fp4, nf4)
    - 🚧 Int8 (https://arxiv.org/abs/2208.07339) support is coming soon!
  - `GGUF` (2-8 bit quantization)
- Easy: Strong support for running 🤗 DDUF models.
- Many devices: NVIDIA GPUs (CUDA), Apple M-series GPUs (Metal), CPU SIMD

## Installation
Check out the [installation guide](INSTALL.md) for details about installation.

## Examples
After [installing](#installation), you can try out these examples!

> Download the DDUF file here: `wget https://huggingface.co/DDUF/FLUX.1-dev-DDUF/resolve/main/FLUX.1-dev-Q4-bnb.dduf`

**CLI:**
```bash
diffuse_rs_cli --scale 3.5 --num-steps 50 dduf -f FLUX.1-dev-Q4-bnb.dduf
```

**Python:**
```py
from diffuse_rs import DiffusionGenerationParams, ModelSource, Pipeline
from PIL import Image
import io

pipeline = Pipeline(source=ModelSource.DdufFile("FLUX.1-dev-Q4-bnb.dduf"))

image_bytes = pipeline.forward(
    prompts=["Draw a picture of a sunrise."],
    params=DiffusionGenerationParams(
        height=720, width=1280, num_steps=50, guidance_scale=3.5
    ),
)

image = Image.open(io.BytesIO(image_bytes[0]))
image.show()
```

**Rust crate:**

Examples with the Rust crate: [here](diffuse_rs_examples/examples).

## Support matrix
| Model | Supports DDUF | Supports quantized DDUF |
| -- | -- | -- |
| FLUX.1 Dev/Schnell | ✅ | ✅ |
