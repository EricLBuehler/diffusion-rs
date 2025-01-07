# Feature flags

Diffuse-rs controls building with GPU support or CPU SIMD acceleration with feature flags.

These are set at compile time and are as follows:

|Feature|Flag|
|--|--|
|Nvidia GPUs (CUDA)|`--features cuda`|
|Apple Silicon GPUs (Metal)|`--features metal`|
|Apple Accelerate (CPU)|`--features accelerate`|
|Intel MKL (CPU)|`--features mkl`|
|Use AVX or NEON automatically|None specified|