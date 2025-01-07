# `diffusion_rs_cli`

CLI for diffusion-rs.

## Examples
- FLUX dev:
```
diffusion_rs_cli --scale 3.5 --num-steps 50 dduf -f FLUX.1-dev-Q4-bnb.dduf
```

```
diffusion_rs_cli --scale 3.5 --num-steps 50 model-id -m black-forest-labs/FLUX.1-dev
```

- FLUX schnell:
```
diffusion_rs_cli --scale 0.0 --num-steps 4 dduf -f FLUX.1-schnell-Q8-bnb.dduf
```

```
diffusion_rs_cli --scale 0.0 --num-steps 4 model-id -m black-forest-labs/FLUX.1-dev
```