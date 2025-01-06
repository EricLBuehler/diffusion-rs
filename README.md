<a name="top"></a>
<h1 align="center">
  diffuse-rs
</h1>

<h3 align="center">
Blazingly fast inference of diffusion models.
</h3>

## Features
- Quantization
  - `bitsandbytes` format (fp4, nf4, and int8)
  - `GGUF` (2-8 bit quantization)
- Easy: Strong support for running [🤗 DDUF](https://huggingface.co/DDUF) models.
- Strong Apple Silicon support: support for the Metal, Accelerate, and ARM NEON frameworks
- Support for NVIDIA GPUs with CUDA
- AVX support for x86 CPUs
- Allow acceleration of models larger than the total VRAM size with offloading

## Upcoming features
- 🚧 LoRA support
- 🚧 CPU + GPU inference with automatic offloading to allow partial acceleration of models larger than the total VRAM

## Installation
Check out the [installation guide](INSTALL.md) for details about installation.

## Examples
After [installing](#installation), you can try out these examples!

> Download the DDUF file here: `wget https://huggingface.co/DDUF/FLUX.1-dev-DDUF/resolve/main/FLUX.1-dev-Q4-bnb.dduf`

**CLI:**
```bash
diffuse_rs_cli --scale 3.5 --num-steps 50 dduf -f FLUX.1-dev-Q4-bnb.dduf
```

More CLI examples [here](diffuse_rs_cli/README.md).

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

## Contributing

- Anyone is welcome to contribute by opening PRs
  - See [good first issues](https://github.com/EricLBuehler/diffuse-rs/labels/good%20first%20issue) for a starting point!
- Collaborators will be invited based on past contributions
