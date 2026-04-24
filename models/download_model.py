#!/usr/bin/env python3
"""
Download GGUF models from Hugging Face for the black-box llama.cpp profiling project.

Usage:
    python download_model.py <repo_id> [filename]

Examples:
    # List available GGUF files in the repo:
    python download_model.py bartowski/Qwen2.5-1.5B-Instruct-GGUF

    # Download a specific GGUF file:
    python download_model.py bartowski/Qwen2.5-1.5B-Instruct-GGUF Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
"""

import argparse
import sys
import os
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, list_repo_files
    from huggingface_hub.utils import (
        EntryNotFoundError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )
except ImportError:
    print("ERROR: huggingface_hub not installed.", file=sys.stderr)
    print("Install with: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)


DEFAULT_OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))+ "/models"


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"


def list_gguf_files(repo_id: str) -> list[str]:
    """Return all .gguf files in a repo, sorted."""
    files = list_repo_files(repo_id)
    return sorted(f for f in files if f.lower().endswith(".gguf"))


def print_no_gguf_error(repo_id: str) -> None:
    model_name = repo_id.split("/")[-1]
    print(
        f"\n[!] No .gguf files found in '{repo_id}'.\n"
        f"\n    llama.cpp only supports GGUF format.\n"
        f"    Look for a repo that provides GGUF quantizations, e.g.:\n"
        f"      - bartowski/<model>-GGUF\n"
        f"      - TheBloke/<model>-GGUF\n"
        f"      - <original-author>/<model>-GGUF\n"
        f"\n    Search: https://huggingface.co/models?search={model_name}+gguf",
        file=sys.stderr,
    )


def print_file_list(repo_id: str, files: list[str]) -> None:
    print(f"\n[i] GGUF files available in {repo_id}:\n")
    for f in files:
        print(f"    {f}")
    print(
        f"\n    To download one, run:\n"
        f"      python download_model.py {repo_id} <filename>\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download GGUF models from Hugging Face for llama.cpp.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "repo_id",
        help="HF repo ID, e.g. 'bartowski/Qwen2.5-1.5B-Instruct-GGUF'",
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default=None,
        help="Specific .gguf file to download. If omitted, lists available files.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save the model (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token for gated/private repos (or set HF_TOKEN env var).",
    )

    args = parser.parse_args()

    if args.token:
        import os
        os.environ["HF_TOKEN"] = args.token

    # Validate filename extension up front
    if args.filename and not args.filename.lower().endswith(".gguf"):
        print(
            f"\n[!] '{args.filename}' is not a .gguf file.\n"
            f"\n    llama.cpp only works with GGUF-format models.\n"
            f"    Run without a filename to see which .gguf files this repo has:\n"
            f"      python download_model.py {args.repo_id}",
            file=sys.stderr,
        )
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Look up what's in the repo
    try:
        gguf_files = list_gguf_files(args.repo_id)
    except RepositoryNotFoundError:
        print(
            f"\n[!] Repo '{args.repo_id}' not found.\n"
            f"    Check spelling, or use --token if it is gated/private.",
            file=sys.stderr,
        )
        return 2
    except HfHubHTTPError as e:
        print(f"\n[!] HF API error: {e}", file=sys.stderr)
        return 3

    if not gguf_files:
        print_no_gguf_error(args.repo_id)
        return 1

    # No filename -> list mode
    if not args.filename:
        print_file_list(args.repo_id, gguf_files)
        return 0

    # Filename given but not in repo
    if args.filename not in gguf_files:
        print(
            f"\n[!] '{args.filename}' not found in {args.repo_id}.",
            file=sys.stderr,
        )
        print_file_list(args.repo_id, gguf_files)
        return 1

    # Download
    target_dir = args.output_dir / args.repo_id.replace("/", "_")
    print(f"[+] Downloading {args.filename}")
    print(f"    from: {args.repo_id}")
    print(f"    to:   {target_dir}")

    try:
        local_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            local_dir=target_dir,
        )
    except EntryNotFoundError:
        print(f"\n[!] File '{args.filename}' not found in repo.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.", file=sys.stderr)
        return 130

    local_path = Path(local_path)
    size = local_path.stat().st_size
    print(f"\n[✓] Done: {local_path}")
    print(f"    Size: {human_size(size)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())