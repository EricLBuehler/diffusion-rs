# Installation guide for diffuse-rs

**ToC**
- [CLI](#cli)
- [Python bindings](#python-bindings)
- [Rust crate](#rust-crate)

## CLI
1) Installing diffuse-rs via the CLI requires a few prerequisites:
    - Install the Rust programming language
        - Follow the instructions on this site: https://rustup.rs/
    - (*Linux/Mac only*) Install OpenSSL (*Ubuntu:* `sudo apt install libssl-dev`, *Brew:* `brew install openssl`)
    - (*Linux only*) Install pkg-config (*Ubuntu:* `sudo apt install pkg-config`)

2) Some models on Hugging Face are gated and require a token to access them:
    - Install the necessary tool: `pip install huggingface_hub`
    - Login: `huggingface_cli login`

3) Clone the repository
```
git clone https://github.com/EricLBuehler/diffuse-rs.git
cd diffuse-rs
```

4) Install the `diffuse_rs_cli` CLI

> [!NOTE]
> Replace the `...` below with [feature flags](FEATURE_FLAGS.md) to build for Nvidia GPUs (CUDA) or Apple Silicon GPUs (Metal)

```
cargo install --path diffuse_rs_cli --release --features ...
```

5) Try the CLI!

> Download the DDUF file here: `wget https://huggingface.co/DDUF/FLUX.1-dev-DDUF/resolve/main/FLUX.1-dev-Q4-bnb.dduf`

```
diffuse_rs_cli --scale 3.5 --num-steps 50 dduf -f FLUX.1-dev-Q4-bnb.dduf
```

## Python bindings
1) Installing diffuse-rs via the Python bindings requires a few prerequisites:
    - Install the Rust programming language
        - Follow the instructions on this site: https://rustup.rs/
    - (*Linux/Mac only*) Install OpenSSL (*Ubuntu:* `sudo apt install libssl-dev`, *Brew:* `brew install openssl`)
    - (*Linux only*) Install pkg-config (*Ubuntu:* `sudo apt install pkg-config`)

2) Some models on Hugging Face are gated and require a token to access them:
    - Install the necessary tool: `pip install huggingface_hub`
    - Login: `huggingface_cli login`

3) Clone the repository
```
git clone https://github.com/EricLBuehler/diffuse-rs.git
cd diffuse-rs
```

4) Install the maturin build tool
```
pip install maturin
```

5) Install the Python bindings

> [!NOTE]
> Replace the `...` below with [feature flags](FEATURE_FLAGS.md) to build for Nvidia GPUs (CUDA) or Apple Silicon GPUs (Metal)

```
maturin develop -m diffuse_rs_py/Cargo.toml --release --features ...
``` 

6) Try the Python bindings!

> Download the DDUF file here: `wget https://huggingface.co/DDUF/FLUX.1-dev-DDUF/resolve/main/FLUX.1-dev-Q4-bnb.dduf`

```py
from diffuse_rs import DiffusionGenerationParams, ModelSource, Pipeline
from PIL import Image
import io

pipeline = Pipeline(source=ModelSource.DdufFile("FLUX.1-dev-Q4-bnb.dduf"))

image_bytes = pipeline.forward(
    prompts=["Draw a picture of a sunrise."],
    params=DiffusionGenerationParams(
        height=720, width=1280, num_steps=50, guidance_scale=3.5
    ),
)

image = Image.open(io.BytesIO(image_bytes[0]))
image.show()
```

## Rust crate
1) Installing diffuse-rs for usage as a Rust crate requires a few prerequisites:
    - Install the Rust programming language
        - Follow the instructions on this site: https://rustup.rs/
    - (*Linux/Mac only*) Install OpenSSL (*Ubuntu:* `sudo apt install libssl-dev`, *Brew:* `brew install openssl`)
    - (*Linux only*) Install pkg-config (*Ubuntu:* `sudo apt install pkg-config`)

2) Some models on Hugging Face are gated and require a token to access them:
    - Install the necessary tool: `pip install huggingface_hub`
    - Login: `huggingface_cli login`

3) Add this dependency to your `Cargo.toml`
    - `diffuse_rs_core = { git = "https://github.com/EricLBuehler/diffuse-rs.git" }`
