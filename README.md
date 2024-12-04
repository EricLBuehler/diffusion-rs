<a name="top"></a>
<h1 align="center">
  diffusers-rs
</h1>

<h3 align="center">
Blazingly fast inference of diffusion models.
</h3>

## Features
- Quantization: using advanced [`mistralrs_quant`](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs-quant) package, we support inference of models:
  - With native accelerator support for every method on CUDA, Metal, and CPU
  - `bitsandbytes` format (int8, fp4, nf4)
  - `GGUF` (2-8 bit quantization, with imatrix)
  - `GPTQ` (with CUDA marlin kernel)
  - `HQQ` (4, 8 bit quantization)
  - `FP8` (optimized on CUDA)