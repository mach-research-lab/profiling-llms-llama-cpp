# Models

This folder stores GGUF model files used for llama.cpp profiling.

> ⚠️ **Don't commit downloaded models to git!** Add this folder (or at least `*.gguf`) to `.gitignore` before downloading anything.

## Setup

Install the Hugging Face Hub client. A virtual environment is recommended (on Ubuntu/Debian, installing into the system Python is blocked by PEP 668):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install huggingface_hub
```

If you'd rather skip the venv, you can use `pipx install huggingface_hub` or `pip install --break-system-packages huggingface_hub`.

## Usage

The script takes a Hugging Face repo ID and a GGUF filename.

### 1. List available GGUF files in a repo

```bash
python3 download_model.py bartowski/Qwen2.5-1.5B-Instruct-GGUF
```

### 2. Download a specific GGUF file

```bash
python3 download_model.py bartowski/Qwen2.5-1.5B-Instruct-GGUF Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
```

### Options

| Flag | Description |
|------|-------------|
| `--output-dir`, `-o` | Where to save the model (default: `~/models`) |
| `--token` | HF token for gated/private repos (or set `HF_TOKEN` env var) |

Example with a custom output directory:

```bash
python3 download_model.py bartowski/Qwen2.5-1.5B-Instruct-GGUF Qwen2.5-1.5B-Instruct-Q4_K_M.gguf -o ./models
```

## Notes

- llama.cpp only supports **GGUF** format. The script rejects non-`.gguf` files and warns you if a repo has no GGUF files, pointing to `bartowski/*-GGUF` or `TheBloke/*-GGUF` as common sources of quantized variants.
- Downloads resume automatically if interrupted.
- Files are saved under `<output-dir>/<repo_owner>_<repo_name>/`.