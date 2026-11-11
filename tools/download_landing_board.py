"""Download (a subset of) the beehive landing-board dataset from Mendeley Data.

Dataset: T. Sledevič et al., "Labeled dataset for bee detection and direction
estimation on beehive landing boards", Mendeley Data, v6, 2024.
DOI 10.17632/8gb9r2yhfc.6 - licence CC BY 4.0.

The full dataset is ~7 GB of 1920x1080 JPEGs. To save disk and bandwidth we
download only ``--per-hive`` frames per hive and store them resized to
``--width`` pixels wide (labels are normalised, so they stay valid).

Every file is checked against the SHA-256 published by Mendeley. Text files
from the archive are only ever *parsed*, never executed.

Usage:
    python tools/download_landing_board.py --part detection --per-hive 250
    python tools/download_landing_board.py --part tracking
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://data.mendeley.com/public-api/datasets/8gb9r2yhfc"
VERSION = 6
UA = {"User-Agent": "smart-hive-dataset-downloader/0.1"}

# Folder ids of the v6 release (from the public folder listing).
DETECTION_HIVES = {
    "20230609a": ("28b86c0e-e562-4111-bece-cce42c8c77d4", "0cf4df30-7033-4c9c-ab2b-bedacdf14047"),
    "20230609b": ("5dbb7244-6e4d-4676-bca8-48943e0e1843", "a0b033b7-8431-4a12-ae0d-0c65345d1175"),
    "20230609c": ("ec37edf4-3456-432f-bb4d-2e265b3223c5", "4b8b5c59-4aac-47ae-9883-e94c4f6c7089"),
    "20230609d": ("ca9e8ad0-f09d-46bf-9009-5e466acee44e", "97ebac51-6f50-4161-a40e-90f24218a02d"),
    "20230609e": ("ce1251d9-b424-4b55-b752-c658fa685d53", "c52352f8-13ad-4e3f-a362-d2e62ca6ff3c"),
    "20230711a": ("1af6ac2a-27be-47c4-858d-877b678f589d", "7b6ae4ce-171c-4ba3-a788-53601ecec59f"),
    "20230711b": ("d7f492c0-f579-4124-bfc5-809c7ba64580", "918fc761-29a1-4cc8-8fe0-ee23ceb70cfd"),
    "20230711c": ("deef0721-b810-43eb-824c-84d425afac81", "e4bbf1a5-8d03-414a-8ea7-7390b7244741"),
}
TRACKING_FOLDER = "ec0c69d5-70d6-453b-a3e2-44619d994ec6"


def _get(url: str, timeout: int = 300) -> bytes:
    last: Exception | None = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as exc:  # network is flaky: retry with backoff
            last = exc
            time.sleep(2 + 4 * attempt)
    raise RuntimeError(f"failed to download {url}: {last}")


def list_folder(folder_id: str) -> list[dict]:
    url = f"{API}/files?folder_id={folder_id}&version={VERSION}"
    return json.loads(_get(url, timeout=120))


def _fetch_checked(entry: dict, tries: int = 4) -> bytes:
    want = entry["content_details"].get("sha256_hash")
    for _ in range(tries):
        data = _get(entry["content_details"]["download_url"])
        if not want or hashlib.sha256(data).hexdigest() == want:
            return data
        time.sleep(3)  # truncated transfer: try again
    raise RuntimeError(f"checksum mismatch for {entry['filename']}")


def _save_resized(data: bytes, dest: Path, width: int) -> None:
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"not a decodable image: {dest.name}")
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def download_detection(out: Path, per_hive: int, width: int, seed: int, workers: int) -> None:
    rng = random.Random(seed)
    for hive, (img_folder, lab_folder) in DETECTION_HIVES.items():
        img_dir = out / "detection" / hive / "images"
        lab_dir = out / "detection" / hive / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(list_folder(img_folder), key=lambda e: e["filename"])
        labels = {Path(e["filename"]).stem: e for e in list_folder(lab_folder)}
        images = [e for e in images if Path(e["filename"]).stem in labels]
        chosen = sorted(rng.sample(images, min(per_hive, len(images))), key=lambda e: e["filename"])

        def job(entry: dict, img_dir: Path = img_dir, lab_dir: Path = lab_dir, labels: dict = labels) -> None:
            stem = Path(entry["filename"]).stem
            dest = img_dir / f"{stem}.jpg"
            if dest.exists() and (lab_dir / f"{stem}.txt").exists():
                return
            try:
                label = _fetch_checked(labels[stem])
                _save_resized(_fetch_checked(entry), dest, width)
                (lab_dir / f"{stem}.txt").write_bytes(label)
            except RuntimeError as exc:  # skip this frame, keep the rest
                dest.unlink(missing_ok=True)
                print(f"skipped {stem}: {exc}", flush=True)

        with ThreadPoolExecutor(workers) as pool:
            for i, _ in enumerate(pool.map(job, chosen), 1):
                if i % 50 == 0:
                    print(f"{hive}: {i}/{len(chosen)}", flush=True)
        print(f"{hive}: done ({len(chosen)} frames)", flush=True)


def download_tracking(out: Path, skip_large: bool) -> None:
    dest_dir = out / "tracking"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for entry in list_folder(TRACKING_FOLDER):
        name = entry["filename"]
        dest = dest_dir / name
        if dest.exists() and dest.stat().st_size == entry["size"]:
            continue
        if skip_large and entry["size"] > 150_000_000:
            print(f"skip {name} ({entry['size'] / 1e6:.0f} MB)")
            continue
        print(f"get {name} ({entry['size'] / 1e6:.1f} MB)", flush=True)
        dest.write_bytes(_fetch_checked(entry))
    print("tracking: done", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["detection", "tracking", "all"], default="all")
    ap.add_argument("--out", type=Path, default=Path("data/raw/landing_board"))
    ap.add_argument("--per-hive", type=int, default=250)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-large", action="store_true", help="skip videos larger than 150 MB")
    args = ap.parse_args(argv)
    if args.part in ("tracking", "all"):
        download_tracking(args.out, args.skip_large)
    if args.part in ("detection", "all"):
        download_detection(args.out, args.per_hive, args.width, args.seed, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
