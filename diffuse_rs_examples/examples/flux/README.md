# FLUX.1 Dev & Schnell

This is an example of inference of FLUX.1 Dev/Schnell.

```
cargo run --release --features metal -- --which dev --prompt "Draw a picture of a beautiful sunset in the winter in the mountains, 4k, high quality."
```

- GPU usage is determined by the `--features` flag: replace `--features metal` with `--features cuda` to use an NVIDIA GPU.