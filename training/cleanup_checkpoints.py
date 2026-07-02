#!/usr/bin/env python3
"""
Clean up old checkpoints across all training runs.

For each run's checkpoints/ directory, keeps only the checkpoint with the
highest epoch number and deletes the rest.

Usage:
    python training/cleanup_checkpoints.py                        # dry-run (default)
    python training/cleanup_checkpoints.py --delete               # actually delete
    python training/cleanup_checkpoints.py --delete --base-dir /custom/path
"""

import argparse
import glob
import os
import re


def find_checkpoint_dirs(base_dir: str) -> list[str]:
    """Find all checkpoints/ directories under base_dir/**/logs/**/checkpoints."""
    pattern = os.path.join(base_dir, "**/logs/**/checkpoints")
    return sorted(glob.glob(pattern, recursive=True))


def parse_epoch(filename: str) -> int | None:
    """Extract epoch number from a checkpoint filename like model_epoch_500.pt."""
    m = re.search(r'(\d+)', filename)
    return int(m.group(1)) if m else None


def cleanup_dir(ckpt_dir: str, delete: bool) -> tuple[int, int, int]:
    """
    Remove all but the highest-epoch .pt file in ckpt_dir.
    Returns (num_kept, num_removed, bytes_freed).
    """
    files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
    if len(files) <= 1:
        return len(files), 0, 0

    # Sort by epoch number descending
    files_with_epoch = [(f, parse_epoch(f)) for f in files]
    files_with_epoch = [(f, e) for f, e in files_with_epoch if e is not None]
    files_with_epoch.sort(key=lambda x: x[1], reverse=True)

    keep = files_with_epoch[0]
    to_remove = files_with_epoch[1:]
    bytes_freed = 0

    if to_remove:
        print(f"{ckpt_dir}/")

    for fname, epoch in to_remove:
        path = os.path.join(ckpt_dir, fname)
        size_bytes = os.path.getsize(path)
        size_mb = size_bytes / (1024 * 1024)
        bytes_freed += size_bytes
        if delete:
            os.remove(path)
            print(f"  DELETED: {path}  ({size_mb:.1f} MB)")
        else:
            print(f"  Would delete: {path}  ({size_mb:.1f} MB)")

    return 1, len(to_remove), bytes_freed


def main():
    parser = argparse.ArgumentParser(description="Clean up old training checkpoints")
    parser.add_argument(
        "--base-dir", default="/home/sazhang/Neural-Radiosity-Renderer",
        help="Root directory to search (default: %(default)s)"
    )
    parser.add_argument(
        "--delete", action="store_true",
        help="Actually delete files (default is dry-run)"
    )
    args = parser.parse_args()

    if not args.delete:
        print("DRY RUN — pass --delete to actually remove files\n")

    ckpt_dirs = find_checkpoint_dirs(args.base_dir)
    print(f"Found {len(ckpt_dirs)} checkpoint directories\n")

    total_kept, total_removed, total_bytes = 0, 0, 0
    for d in ckpt_dirs:
        kept, removed, freed = cleanup_dir(d, args.delete)
        total_kept += kept
        total_removed += removed
        total_bytes += freed

    action = 'Freed' if args.delete else 'Would free'
    if total_bytes >= 1024 ** 3:
        size_str = f"{total_bytes / (1024 ** 3):.2f} GB"
    else:
        size_str = f"{total_bytes / (1024 ** 2):.1f} MB"
    print(f"\nSummary: kept {total_kept}, {'deleted' if args.delete else 'would delete'} {total_removed}")
    print(f"{action}: {size_str}")


if __name__ == "__main__":
    main()
