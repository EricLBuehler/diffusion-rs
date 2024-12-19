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
- Many devices: NVIDIA GPUs (CUDA), Apple M-series GPUs (Metal)

## Installation
Check out the [installation guide](INSTALL.md) for details about installation.

## Examples
After [installing](#installation), you can try out these examples!

- Examples with the Rust crate: [here](diffusers_examples/examples).

**CLI:**
```bash
> diffusers_cli --scale 3.5 --num-steps 50 dduf -f FLUX.1-dev-Q4-bnb.dduf
```

## Support matrix
| Model | Supports quantization |
| -- | -- |
| FLUX.1 Dev/Schnell | ✅ |
