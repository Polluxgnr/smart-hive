"""Evaluate the detector (mAP on held-out data) and the crop classifier (per-class P/R).

Detection is reported separately for three test subsets, because they answer
different questions:
    * landing-board frames of the 3 held-out hives  -> bees at real entrances
    * held-out iNaturalist photos                   -> vespids in photos
    * held-out composites (vespids pasted on held-out hives) -> vespids at entrance scale

Writes docs/results/models.json and figures in docs/img/.

Usage:
    python tools/eval_models.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CLASSES = ["bee", "bee_pollen", "vespa_velutina", "vespa_crabro", "wasp"]


def detection(args) -> dict:
    from ultralytics import YOLO

    root = args.data / "detect"
    test_imgs = sorted((root / "images" / "test").glob("*.jpg"))
    subsets = {
        "landing_board_test_hives": [p for p in test_imgs if not p.name.startswith(("inat_", "comp_"))],
        "inat_test_photos": [p for p in test_imgs if p.name.startswith("inat_")],
        "composites_test_hives": [p for p in test_imgs if p.name.startswith("comp_")],
    }
    model = YOLO(args.detector)
    out = {}
    for name, files in subsets.items():
        if not files:
            continue
        lst = root / f"test_{name}.txt"
        lst.write_text("\n".join(str(p.resolve()) for p in files), encoding="utf-8")
        yaml = root / f"test_{name}.yaml"
        yaml.write_text(
            f"path: {root.resolve().as_posix()}\ntrain: images/train\nval: {lst.resolve().as_posix()}\n"
            "names:\n  0: bee\n  1: vespid\n",
            encoding="utf-8",
        )
        m = model.val(
            data=str(yaml),
            imgsz=args.imgsz,
            batch=8,
            device="cpu",
            plots=False,
            verbose=False,
            project=str(Path("runs").resolve()),
            name=f"val_{name}",
            exist_ok=True,
        )
        per_class = {}
        for i, c in enumerate(m.box.ap_class_index):
            per_class[model.names[int(c)]] = {
                "precision": float(m.box.p[i]),
                "recall": float(m.box.r[i]),
                "map50": float(m.box.ap50[i]),
                "map50_95": float(m.box.ap[i]),
            }
        out[name] = {
            "images": len(files),
            "map50": float(m.box.map50),
            "map50_95": float(m.box.map),
            "per_class": per_class,
            "speed_ms": m.speed,
        }
        print(name, json.dumps(out[name]), flush=True)
    return out


def classification(args) -> dict:
    from ultralytics import YOLO

    model = YOLO(args.classifier)
    names = {int(k): v for k, v in model.names.items()}
    idx = {v: k for k, v in names.items()}
    root = args.data / "classify" / "test"
    present = [c for c in CLASSES if (root / c).exists()]
    conf = np.zeros((len(present), len(present)), int)
    for i, c in enumerate(present):
        files = sorted((root / c).glob("*.jpg"))
        for k in range(0, len(files), 64):
            for r in model.predict([str(f) for f in files[k : k + 64]], imgsz=96, device="cpu", verbose=False):
                pred = names[int(r.probs.top1)]
                if pred in present:
                    conf[i, present.index(pred)] += 1
    per_class = {}
    for i, c in enumerate(present):
        tp = conf[i, i]
        per_class[c] = {
            "n": int(conf[i].sum()),
            "precision": float(tp / conf[:, i].sum()) if conf[:, i].sum() else None,
            "recall": float(tp / conf[i].sum()) if conf[i].sum() else None,
        }
    _ = idx
    out = {
        "classes": present,
        "confusion": conf.tolist(),
        "per_class": per_class,
        "accuracy": float(np.trace(conf) / conf.sum()),
    }
    plot_confusion(conf, present, args.figdir / "confusion_species.png")
    print(json.dumps(out, indent=1))
    return out


def plot_confusion(conf: np.ndarray, labels: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=150)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    pretty = [lb.replace("vespa_", "V. ").replace("_", " ") for lb in labels]
    ax.set_xticks(range(len(labels)), pretty, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), pretty)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{conf[i, j]}",
                ha="center",
                va="center",
                color="white" if norm[i, j] > 0.5 else "#0b0b0b",
                fontsize=9,
            )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Species classifier, held-out crops (counts)")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/datasets"))
    ap.add_argument("--detector", default="models/detector.pt")
    ap.add_argument("--classifier", default="models/classifier.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--figdir", type=Path, default=Path("docs/img"))
    ap.add_argument("--out", type=Path, default=Path("docs/results/models.json"))
    ap.add_argument("--only", choices=["detect", "classify"])
    args = ap.parse_args(argv)
    res = json.loads(args.out.read_text()) if args.out.exists() else {}
    if args.only in (None, "detect"):
        res["detection"] = detection(args)
    if args.only in (None, "classify"):
        res["classification"] = classification(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
