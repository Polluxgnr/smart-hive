"""Download openly licensed iNaturalist photos of hornets, wasps and honey bees.

Only research-grade observations whose photo licence is CC0, CC BY or CC BY-SA
are kept, and an ``attribution.csv`` records the observation, photographer and
licence of every image (CC BY requires credit).

Classes (iNaturalist taxon ids, European records only):
    vespa_velutina  119019  Asian (yellow-legged) hornet - invasive in France
    vespa_crabro     54327  European hornet - native, protected in some regions
    vespula          61356  common / German wasps
    polistes         52756  paper wasps
    apis_mellifera   47219  Western honey bee

The API is queried at most once per second, as iNaturalist asks.

Usage:
    python tools/download_inaturalist.py --out data/raw/inat --per-class 600
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

API = "https://api.inaturalist.org/v1/observations"
UA = {"User-Agent": "smart-hive-dataset-downloader/0.1"}
EUROPE = 97391
LICENCES = "cc0,cc-by,cc-by-sa"
TAXA = {
    "vespa_velutina": 119019,
    "vespa_crabro": 54327,
    "vespula": 61356,
    "polistes": 52756,
    "apis_mellifera": 47219,
}


def _get(url: str, timeout: int = 120) -> bytes:
    last: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception as exc:
            last = exc
            time.sleep(3 + 5 * attempt)
    raise RuntimeError(f"failed: {url}: {last}")


def fetch_class(name: str, taxon: int, n: int, out: Path, writer: Any) -> None:
    dest = out / name
    dest.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in dest.glob("*.jpg")}
    got = len(have)
    id_above = 0
    while got < n:
        q = {
            "taxon_id": taxon,
            "place_id": EUROPE,
            "quality_grade": "research",
            "photo_license": LICENCES,
            "per_page": 200,
            "order_by": "id",
            "order": "asc",
            "id_above": id_above,
            "photos": "true",
        }
        page = json.loads(_get(f"{API}?{urllib.parse.urlencode(q)}"))
        time.sleep(1.0)
        results = page.get("results", [])
        if not results:
            break
        for obs in results:
            id_above = max(id_above, obs["id"])
            # One photo per observation keeps the set diverse.
            photos = [p for p in obs.get("photos", []) if (p.get("license_code") or "") in LICENCES.split(",")]
            if not photos:
                continue
            photo = photos[0]
            stem = f"{name}_{photo['id']}"
            if stem in have:
                continue
            url = photo["url"].replace("/square.", "/medium.")
            try:
                (dest / f"{stem}.jpg").write_bytes(_get(url))
            except RuntimeError as exc:
                print(exc, file=sys.stderr)
                continue
            writer.writerow(
                [
                    f"{name}/{stem}.jpg",
                    obs["id"],
                    photo["id"],
                    photo["license_code"],
                    photo.get("attribution", ""),
                    obs.get("uri", ""),
                ]
            )
            have.add(stem)
            got += 1
            if got % 100 == 0:
                print(f"{name}: {got}/{n}", flush=True)
            if got >= n:
                break
    print(f"{name}: done ({got})", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/raw/inat"))
    ap.add_argument("--per-class", type=int, default=600)
    ap.add_argument("--classes", nargs="*", default=list(TAXA))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    attribution = args.out / "attribution.csv"
    new = not attribution.exists()
    with attribution.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(["file", "observation_id", "photo_id", "licence", "attribution", "observation_url"])
        for name in args.classes:
            fetch_class(name, TAXA[name], args.per_class, args.out, writer)
            fh.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
