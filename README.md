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
- Easy: Load models directly from the [🤗 diffusers](https://github.com/huggingface/diffusers) multifolder format

## Examples
- Examples with the Rust crate: [here](diffusers_examples/examples).

## Support matrix
- FLUX.1 dev and FLUX.1 schnell