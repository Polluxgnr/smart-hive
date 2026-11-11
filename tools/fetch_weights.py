"""Download the trained models listed in models/MANIFEST.json and check their SHA-256.

The weights are not stored in git (see .gitignore). They are attached to a
GitHub release; to rebuild them from the public datasets instead, see the
"Reproduce" section of docs/report.md.

Usage:
    python tools/fetch_weights.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "models" / "MANIFEST.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ok = True
    for name, entry in manifest["files"].items():
        dest = ROOT / "models" / name
        if dest.exists() and sha256(dest) == entry["sha256"]:
            print(f"{name}: already present")
            continue
        url = f"{manifest['base_url'].rstrip('/')}/{name}"
        print(f"{name}: downloading {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "smart-hive/0.1"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
        except Exception as exc:
            print(f"  failed: {exc}\n  -> train it yourself: see docs/report.md, 'Reproduce'")
            ok = False
            continue
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(data)
        if sha256(tmp) != entry["sha256"]:
            tmp.unlink()
            print("  checksum mismatch: file rejected")
            ok = False
            continue
        tmp.replace(dest)
        print(f"  ok ({len(data) / 1e6:.1f} MB)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
