# `diffuse_rs_cli`

CLI for diffuse-rs.

## Examples
- FLUX dev:
```
diffuse_rs_cli --scale 3.5 --num-steps 50 dduf -f FLUX.1-dev-Q4-bnb.dduf
```

```
diffuse_rs_cli --scale 3.5 --num-steps 50 model-id -m black-forest-labs/FLUX.1-dev
```

- FLUX schnell:
```
diffuse_rs_cli --scale 0.0 --num-steps 4 dduf -f FLUX.1-schnell-Q8-bnb.dduf
```

```
diffuse_rs_cli --scale 0.0 --num-steps 4 model-id -m black-forest-labs/FLUX.1-dev
```