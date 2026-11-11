"""Build the YOLO detection and classification datasets from the raw downloads.

Splits are by **hive** for the landing-board frames (frames of one hive are
near duplicates, so a frame-level split would leak) and by **observation** for
iNaturalist photos. The split is written to data/splits/ and reused by every
evaluation script.

    hives 20230609a, c, d, e   -> train
    hive  20230711c            -> val (model selection)
    hives 20230609b, 20230711a, 20230711b -> test (also the hives of the tracking
                                  videos used for the counting evaluation)

Detection classes: 0 bee, 1 vespid. Vespid boxes come from the pseudo-labels of
tools/pseudo_label_inat.py; in addition, vespid crops are pasted onto
landing-board frames with a soft elliptical mask (copy-paste augmentation) so the
detector sees hornets at entrance scale.

Classification classes: bee, bee_pollen, vespa_velutina, vespa_crabro, wasp.

Usage:
    python tools/build_datasets.py --composites 600
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

TRAIN_HIVES = ["20230609a", "20230609c", "20230609d", "20230609e"]
VAL_HIVES = ["20230711c"]
TEST_HIVES = ["20230609b", "20230711a", "20230711b"]
INAT_TO_CLS = {
    "vespa_velutina": "vespa_velutina",
    "vespa_crabro": "vespa_crabro",
    "vespula": "wasp",
    "polistes": "wasp",
    "apis_mellifera": "bee",
}
VESPID_FOLDERS = ("vespa_velutina", "vespa_crabro", "vespula", "polistes")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def read_yolo(path: Path) -> list[list[float]]:
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                rows.append([float(v) for v in parts[:5]])
    return rows


def write_yolo(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in rows))


def inat_split(inat: Path, boxes: dict, seed: int) -> dict[str, str]:
    """photo -> train/val/test, stratified by folder, 80/10/10."""
    rng = random.Random(seed)
    split: dict[str, str] = {}
    for folder in sorted({k.split("/")[0] for k in boxes}):
        items = sorted(k for k in boxes if k.startswith(folder + "/"))
        rng.shuffle(items)
        n = len(items)
        for i, k in enumerate(items):
            split[k] = "test" if i < n * 0.1 else "val" if i < n * 0.2 else "train"
    return split


def hive_frames(lb: Path, hive: str) -> list[Path]:
    return sorted((lb / "detection" / hive / "images").glob("*.jpg"))


def soft_paste(dst: np.ndarray, crop: np.ndarray, cx: int, cy: int, rng: random.Random) -> tuple[int, int, int, int]:
    """Alpha-blend ``crop`` centred at (cx, cy) with a feathered ellipse. Returns the box."""
    h, w = crop.shape[:2]
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.46), int(h * 0.46)), 0, 0, 360, 1.0, -1)
    k = max(3, (min(h, w) // 6) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    if rng.random() < 0.5:  # flying hornets are often motion-blurred
        kk = rng.choice([3, 5, 7])
        kernel = np.zeros((kk, kk), np.float32)
        kernel[kk // 2, :] = 1.0 / kk
        crop = cv2.filter2D(crop, -1, kernel)
    x1, y1 = cx - w // 2, cy - h // 2
    H, W = dst.shape[:2]
    a1, b1 = max(0, x1), max(0, y1)
    a2, b2 = min(W, x1 + w), min(H, y1 + h)
    if a2 <= a1 or b2 <= b1:
        return 0, 0, 0, 0
    m = mask[b1 - y1 : b2 - y1, a1 - x1 : a2 - x1, None]
    region = dst[b1:b2, a1:a2].astype(np.float32)
    patch = crop[b1 - y1 : b2 - y1, a1 - x1 : a2 - x1].astype(np.float32)
    dst[b1:b2, a1:a2] = (m * patch + (1 - m) * region).astype(np.uint8)
    # Tight box: the ellipse covers ~92 % of the crop.
    return a1 + int(w * 0.06), b1 + int(h * 0.06), a2 - int(w * 0.06), b2 - int(h * 0.06)


def vespid_crop(inat: Path, rel: str, box: list[float], rng: random.Random) -> np.ndarray | None:
    img = cv2.imread(str(inat / rel))
    if img is None:
        return None
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    pad_x, pad_y = int((x2 - x1) * 0.08), int((y2 - y1) * 0.08)
    crop = img[max(0, y1 - pad_y) : y2 + pad_y, max(0, x1 - pad_x) : x2 + pad_x]
    if crop.size == 0 or min(crop.shape[:2]) < 16:
        return None
    if rng.random() < 0.5:
        crop = cv2.flip(crop, 1)
    angle = rng.choice([0, 90, 180, 270])
    if angle:
        crop = np.ascontiguousarray(np.rot90(crop, angle // 90))
    return crop


def build_detection(args, boxes: dict, split: dict) -> None:
    out = args.out / "detect"
    rng = random.Random(args.seed)
    lb = args.landing_board
    counts = {"train": 0, "val": 0, "test": 0}
    for part, hives in (("train", TRAIN_HIVES), ("val", VAL_HIVES), ("test", TEST_HIVES)):
        for hive in hives:
            for img in hive_frames(lb, hive):
                lab = img.parent.parent / "labels" / (img.stem + ".txt")
                rows = [r for r in read_yolo(lab) if r[3] > 0 and r[4] > 0]
                link_or_copy(img, out / "images" / part / img.name)
                write_yolo(out / "labels" / part / (img.stem + ".txt"), [[0, *r[1:]] for r in rows])
                counts[part] += 1
    # iNaturalist photos (vespids as class 1, honey bees as class 0).
    for rel, bxs in boxes.items():
        part = split[rel]
        folder = rel.split("/")[0]
        cls = 1 if folder in VESPID_FOLDERS else 0
        img = args.inat / rel
        im = cv2.imread(str(img))
        if im is None:
            continue
        h, w = im.shape[:2]
        rows = [[cls, (b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h, (b[2] - b[0]) / w, (b[3] - b[1]) / h] for b in bxs]
        name = f"inat_{Path(rel).stem}.jpg"
        link_or_copy(img, out / "images" / part / name)
        write_yolo(out / "labels" / part / f"inat_{Path(rel).stem}.txt", rows)
        counts[part] += 1
    # Copy-paste composites: vespids on landing boards, at 1.6-2.4x bee size.
    for part, hives, n in (
        ("train", TRAIN_HIVES, args.composites),
        ("val", VAL_HIVES, args.composites // 8),
        ("test", TEST_HIVES, args.composites // 4),
    ):
        frames = [f for h in hives for f in hive_frames(lb, h)]
        donors = [(k, b) for k, v in boxes.items() if split[k] == part and k.split("/")[0] in VESPID_FOLDERS for b in v]
        if not frames or not donors:
            continue
        for i in range(n):
            frame_path = rng.choice(frames)
            frame = cv2.imread(str(frame_path))
            lab = frame_path.parent.parent / "labels" / (frame_path.stem + ".txt")
            rows = [[0, *r[1:]] for r in read_yolo(lab) if r[3] > 0 and r[4] > 0]
            H, W = frame.shape[:2]
            bee = float(np.median([max(r[3] * W, r[4] * H) for r in rows])) if rows else 0.04 * W
            for _ in range(rng.choice([1, 1, 2])):
                rel, box = rng.choice(donors)
                crop = vespid_crop(args.inat, rel, box, rng)
                if crop is None:
                    continue
                target = bee * rng.uniform(1.6, 2.4)
                s = target / max(crop.shape[:2])
                crop = cv2.resize(crop, (max(8, int(crop.shape[1] * s)), max(8, int(crop.shape[0] * s))))
                cx, cy = rng.randint(int(0.05 * W), int(0.95 * W)), rng.randint(int(0.05 * H), int(0.95 * H))
                x1, y1, x2, y2 = soft_paste(frame, crop, cx, cy, rng)
                if x2 > x1 and y2 > y1:
                    rows.append([1, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H])
            name = f"comp_{part}_{i:04d}"
            cv2.imwrite(str(out / "images" / part / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            write_yolo(out / "labels" / part / f"{name}.txt", rows)
            counts[part] += 1
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: bee\n  1: vespid\n",
        encoding="utf-8",
    )
    print("detection images:", counts)


def build_classification(args, boxes: dict, split: dict) -> None:
    out = args.out / "classify"
    rng = random.Random(args.seed + 1)
    counts: dict[tuple[str, str], int] = {}

    def save(part: str, cls: str, crop: np.ndarray, name: str) -> None:
        d = out / part / cls
        d.mkdir(parents=True, exist_ok=True)
        side = max(crop.shape[:2])
        if side > 128:
            s = 128 / side
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
        cv2.imwrite(str(d / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        counts[(part, cls)] = counts.get((part, cls), 0) + 1

    # Landing-board bee crops (same square-with-context crop as at run time).
    per_part = {"train": args.bee_crops, "val": args.bee_crops // 5, "test": args.bee_crops // 3}
    for part, hives in (("train", TRAIN_HIVES), ("val", VAL_HIVES), ("test", TEST_HIVES)):
        cand = []
        for hive in hives:
            for img in hive_frames(args.landing_board, hive):
                lab = img.parent.parent / "labels" / (img.stem + ".txt")
                cand += [(img, r) for r in read_yolo(lab) if r[3] > 0 and r[4] > 0]
        rng.shuffle(cand)
        cache: dict[Path, np.ndarray] = {}
        for j, (img, r) in enumerate(cand[: per_part[part]]):
            im = cache.setdefault(img, cv2.imread(str(img)))
            if len(cache) > 64:
                cache.clear()
            H, W = im.shape[:2]
            cx, cy, bw, bh = r[1] * W, r[2] * H, r[3] * W, r[4] * H
            crop = square_crop(im, cx, cy, max(bw, bh) * 1.5)
            save(part, "bee", crop, f"lb_{img.stem}_{j}")
    # iNaturalist crops.
    for rel, bxs in boxes.items():
        cls = INAT_TO_CLS[rel.split("/")[0]]
        im = cv2.imread(str(args.inat / rel))
        if im is None:
            continue
        for j, b in enumerate(bxs):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            crop = square_crop(im, cx, cy, max(b[2] - b[0], b[3] - b[1]) * 1.5)
            save(split[rel], cls, crop, f"inat_{Path(rel).stem}_{j}")
    # Pollen: VnPollenBee crops if they were prepared (tools/prepare_pollen.py).
    pollen_dir = args.out.parent / "interim" / "pollen_crops"
    if pollen_dir.exists():
        for part in ("train", "val", "test"):
            for cls in ("bee", "bee_pollen"):
                for f in sorted((pollen_dir / part / cls).glob("*.jpg")):
                    link_or_copy(f, out / part / cls / f"vn_{f.name}")
                    counts[(part, cls)] = counts.get((part, cls), 0) + 1
    print("classification crops:", dict(sorted(counts.items())))


def filter_boxes(raw: dict, min_conf: float, max_boxes: int) -> dict:
    """Keep boxes >= max(min_conf, half the photo's best score); drop empty or crowded photos."""
    out = {}
    for rel, bxs in raw.items():
        if not bxs:
            continue
        best = max(b[4] for b in bxs)
        keep = [b for b in bxs if b[4] >= max(min_conf, 0.5 * best)]
        if 0 < len(keep) <= max_boxes:
            out[rel] = keep
    return out


def square_crop(im: np.ndarray, cx: float, cy: float, side: float) -> np.ndarray:
    H, W = im.shape[:2]
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c, d = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    return im[b:d, a:c]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--landing-board", type=Path, default=Path("data/raw/landing_board"))
    ap.add_argument("--inat", type=Path, default=Path("data/raw/inat"))
    ap.add_argument("--boxes", type=Path, default=Path("data/interim/inat_boxes.json"))
    ap.add_argument("--out", type=Path, default=Path("data/datasets"))
    ap.add_argument("--composites", type=int, default=600)
    ap.add_argument("--bee-crops", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-conf", type=float, default=0.12, help="pseudo-label score cut")
    ap.add_argument("--max-boxes", type=int, default=3, help="drop photos with more insects (nests)")
    ap.add_argument("--only", choices=["detect", "classify"])
    args = ap.parse_args(argv)
    boxes = filter_boxes(json.loads(args.boxes.read_text(encoding="utf-8")), args.min_conf, args.max_boxes)
    filtered = args.boxes.with_name("inat_boxes_filtered.json")
    filtered.write_text(json.dumps(boxes), encoding="utf-8")
    print(f"{len(boxes)} photos kept after the confidence filter -> {filtered}")
    split = inat_split(args.inat, boxes, args.seed)
    splits_dir = args.out.parent / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "inat_split.json").write_text(json.dumps(split, indent=0), encoding="utf-8")
    (splits_dir / "hives.json").write_text(
        json.dumps({"train": TRAIN_HIVES, "val": VAL_HIVES, "test": TEST_HIVES}, indent=1), encoding="utf-8"
    )
    if args.only in (None, "detect"):
        build_detection(args, boxes, split)
    if args.only in (None, "classify"):
        build_classification(args, boxes, split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
