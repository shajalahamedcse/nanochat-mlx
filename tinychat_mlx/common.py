"""
Common utilities for tinychat-mlx.
"""

import os
import mlx.core as mx


def print0(s="", **kwargs):
    """Print (no DDP, always rank 0)."""
    print(s, **kwargs)


def get_base_dir():
    """Get the base directory for weights, data, and tokenizer artifacts.

    Default: weights/ folder at the project root (next to tinychat_mlx/).
    Override with the NANOCHAT_BASE_DIR environment variable.
    """
    if os.environ.get("NANOCHAT_BASE_DIR"):
        base = os.environ["NANOCHAT_BASE_DIR"]
    else:
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(pkg_dir)
        base = os.path.join(project_root, "weights")
    os.makedirs(base, exist_ok=True)
    return base


def set_memory_limit(gb=16):
    """Cap MLX Metal memory usage."""
    limit = int(gb * 1024**3)
    try:
        mx.set_memory_limit(limit)
    except AttributeError:
        mx.metal.set_memory_limit(limit)
    print0(f"MLX memory limit set to {gb}GB")


def get_active_memory_mb():
    """Get current MLX Metal memory usage in MB."""
    try:
        return mx.get_active_memory() / 1024**2
    except AttributeError:
        try:
            return mx.metal.get_active_memory() / 1024**2
        except Exception:
            return 0.0


def get_peak_memory_mb():
    """Get peak MLX Metal memory usage in MB."""
    try:
        return mx.get_peak_memory() / 1024**2
    except AttributeError:
        try:
            return mx.metal.get_peak_memory() / 1024**2
        except Exception:
            return 0.0


def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a simple check-before-download approach (no DDP, single-device).
    """
    import urllib.request
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)

    if os.path.exists(file_path):
        return file_path

    print(f"Downloading {url}...")
    with urllib.request.urlopen(url) as response:
        content = response.read()

    with open(file_path, 'wb') as f:
        f.write(content)
    print(f"Downloaded to {file_path}")

    if postprocess_fn is not None:
        postprocess_fn(file_path)

    return file_path
