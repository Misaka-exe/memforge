"""Reproducible downloader + manifest writer for LongMemEval-S cleaned data.

Usage:
    python -m benchmarks.scripts.download_longmemeval
    python -m benchmarks.scripts.download_longmemeval --mirror hf-mirror.com
    python -m benchmarks.scripts.download_longmemeval --verify-only   # data exists: verify + manifest

Writes:
    benchmarks/data/longmemeval_s_cleaned.json      (gitignored, ~277 MB)
    benchmarks/data/manifests/longmemeval_s.json    (tracked: dataset, sha256, size, num_questions)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

DATASET = "xiaowu0162/longmemeval-cleaned"
FILENAME = "longmemeval_s_cleaned.json"
EXPECTED_NUM_QUESTIONS = 500
DEFAULT_MIRROR = None  # official huggingface.co

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / FILENAME
MANIFEST_PATH = DATA_DIR / "manifests" / "longmemeval_s.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_url(mirror: str | None) -> str:
    host = mirror or "huggingface.co"
    return f"https://{host}/datasets/{DATASET}/resolve/main/{FILENAME}"


def download(mirror: str | None) -> None:
    url = resolve_url(mirror)
    print(f"[download] {url}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-L", "--fail", "--retry", "2", "-o", str(OUT_PATH), url]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        import urllib.request

        urllib.request.urlretrieve(url, OUT_PATH)  # noqa: S310
    except subprocess.CalledProcessError as e:
        print(f"[download] FAILED (exit {e.returncode}). Try --mirror hf-mirror.com")
        sys.exit(1)
    print(f"[download] done: {OUT_PATH.stat().st_size} bytes")


def verify() -> dict:
    if not OUT_PATH.exists():
        print("[verify] data file missing; run without --verify-only first")
        sys.exit(1)
    import json as _json

    data = _json.load(open(OUT_PATH, encoding="utf-8"))
    num = len(data)
    size = OUT_PATH.stat().st_size
    digest = sha256_of(OUT_PATH)
    manifest = {
        "dataset": "LongMemEval-S",
        "source": DATASET,
        "filename": FILENAME,
        "num_questions": num,
        "sha256": digest,
        "size_bytes": size,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("[verify] num_questions:", num, "(expected", EXPECTED_NUM_QUESTIONS, ")")
    print("[verify] size_bytes:", size)
    print("[verify] sha256:", digest)
    ok = num == EXPECTED_NUM_QUESTIONS
    print("[verify]", "OK" if ok else "MISMATCH (see report)")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval-S downloader + verifier")
    parser.add_argument("--mirror", default=DEFAULT_MIRROR, help="HF mirror host, e.g. hf-mirror.com")
    parser.add_argument("--verify-only", action="store_true", help="skip download, just verify + write manifest")
    args = parser.parse_args()

    if not args.verify_only:
        download(args.mirror)
    verify()


if __name__ == "__main__":
    main()