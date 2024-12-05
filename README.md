<a name="top"></a>
<h1 align="center">
  diffuse-rs
</h1>

<h3 align="center">
Blazingly fast inference of diffusion models.
</h3>

## Roadmap
- [x] Implement loading from the diffusers multifolder format, but converting to old Flux format
- [x] Implement loading from & running with diffusers format
- [x] Support loading quantized models

## Features
- Quantization: using the [`mistralrs_quant`](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs-quant) package, we support inference of models
  - With native accelerator support for every method on CUDA, Metal, and CPU
  - `bitsandbytes` format (int8, fp4, nf4)
  - `GGUF` (2-8 bit quantization, with imatrix)
  - `GPTQ` (with CUDA marlin kernel)
  - `HQQ` (4, 8 bit quantization)
  - `FP8` (optimized on CUDA)
- Easy: Load models directly from the [🤗 diffusers](https://github.com/huggingface/diffusers) multifolder format

## Support matrix
- FLUX.1 dev and FLUX.1 schnell