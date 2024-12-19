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
- Easy: Load models directly from the [🤗 diffuse_rs](https://github.com/huggingface/diffuse_rs) multifolder format
- Many devices: NVIDIA GPUs (CUDA), Apple M-series GPUs (Metal)

## Examples
- Examples with the Rust crate: [here](diffuse_rs_examples/examples).

## Support matrix
| Model | Supports quantization |
| -- | -- |
| FLUX.1 Dev/Schnell | ✅ |
