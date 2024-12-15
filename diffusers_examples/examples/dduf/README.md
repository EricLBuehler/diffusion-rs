# DDUF inference

This is an example of inference of DDUF models. Please find the [collection here](https://huggingface.co/DDUF).

```
wget https://huggingface.co/DDUF/FLUX.1-dev-DDUF/resolve/main/FLUX.1-dev-Q4-bnb.dduf

cargo run --release --features metal -- --which FLUX.1-dev-Q4-bnb.dduf --prompt "Draw a picture of a beautiful sunset in the winter in the mountains, 4k, high quality."
```

- GPU usage is determined by the `--features` flag: replace `--features metal` with `--features cuda` to use an NVIDIA GPU.