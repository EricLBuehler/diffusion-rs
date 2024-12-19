# DDUF inference

This is an example of inference of DDUF models. Please find the [collection here](https://huggingface.co/DDUF).

```
# FLUX Dev
wget https://huggingface.co/DDUF/FLUX.1-dev-DDUF/resolve/main/FLUX.1-dev-Q4-bnb.dduf

cargo run --release --features metal --example dduf -- --file FLUX.1-dev-Q4-bnb.dduf --prompt "Draw a picture of a beautiful sunset in the winter in the mountains, 4k, high quality." --num-steps 50 --guidance-scale 3.5

# FLUX Schnell
wget https://huggingface.co/DDUF/FLUX.1-schnell-DDUF/resolve/main/FLUX.1-dev-Q4-bnb.dduf

cargo run --release --features metal --example dduf -- --file FLUX.1-schnell-Q4-bnb.dduf --prompt "Draw a picture of a beautiful sunset in the winter in the mountains, 4k, high quality." --num-steps 4 --guidance-scale 0.0
```

- GPU usage is determined by the `--features` flag: replace `--features metal` with `--features cuda` to use an NVIDIA GPU.