"""Command line: ``smart-hive demo | run | simulate | status | dashboard | serve | calibrate | test-alert | export``."""

from __future__ import annotations

import argparse
import contextlib
import copy
import html
import json
import logging
import sys
import webbrowser
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

from . import dashboard
from .alerts import Alert, Notifier
from .config import Config, EntranceCfg, load_config
from .simulate import SCENARIOS, simulate
from .status import compute_day, describe, notify_status, recompute_all
from .storage import Storage

log = logging.getLogger("smarthive")


def _cfg(args) -> Config:
    path = getattr(args, "config", None)
    if path is None and Path("config.yaml").exists():
        path = "config.yaml"
    return load_config(path)


# -- simulate ----------------------------------------------------------------
def write_simulation(
    storage: Storage,
    cfg: Config,
    scenario: str,
    days: int,
    seed: int,
    hive_id: str | None = None,
    end: date | None = None,
) -> None:
    hive = hive_id or cfg.hive.id
    storage.clear_hive(hive)
    for sd in simulate(days=days, scenario=scenario, seed=seed, end=end):
        for row in sd.rows:
            storage.add_hour(hive, sd.day, row, simulated=True)
        for hour, duration, conf in sd.hornet_events:
            ts = f"{sd.day}T{hour:02d}:{(int(duration * 7) % 60):02d}:00"
            storage.add_event(hive, ts, "vespa_velutina", conf, duration, simulated=True)
        compute_day(storage, cfg, sd.day, hive, weather=sd.weather)


def cmd_simulate(args) -> int:
    cfg = _cfg(args)
    storage = Storage(args.data_dir or cfg.storage.data_dir)
    write_simulation(storage, cfg, args.scenario, args.days, args.seed)
    last = storage.daily_rows(cfg.hive.id, limit=1)[-1]
    print(f"Simulated {args.days} days of '{args.scenario}' for {cfg.hive.id}. Last day: {last['level']}")
    for r in last["reasons"]:
        print(" -", r)
    page = dashboard.render(storage, cfg, Path(storage.data_dir) / "dashboard")
    print(f"Dashboard: {page}")
    return 0


# -- status / dashboard / serve ----------------------------------------------
def cmd_status(args) -> int:
    cfg = _cfg(args)
    storage = Storage(cfg.storage.data_dir)
    statuses = recompute_all(storage, cfg)
    if not statuses:
        print("No data yet.")
        return 0
    st = statuses[-1]
    print(f"{cfg.hive.name} - {st.day}: {st.level}{' - OPEN THE HIVE' if st.open_hive else ''}")
    for line in describe(st, cfg.language):
        print(" -", line)
    if args.notify:
        notify_status(Notifier(cfg.alerts, dry_run=args.dry_run), cfg, st)
    return 0


def cmd_dashboard(args) -> int:
    cfg = _cfg(args)
    storage = Storage(cfg.storage.data_dir)
    page = dashboard.render(storage, cfg, args.out or Path(cfg.storage.data_dir) / "dashboard")
    print(page)
    return 0


def cmd_serve(args) -> int:
    cfg = _cfg(args)
    directory = Path(args.dir or Path(cfg.storage.data_dir) / "dashboard")
    if not (directory / "index.html").exists():
        storage = Storage(cfg.storage.data_dir)
        dashboard.render(storage, cfg, directory)
    dashboard.serve(directory, args.host or cfg.dashboard.host, args.port or cfg.dashboard.port)
    return 0


# -- run (live camera or a video file) ----------------------------------------
def _perception(cfg: Config):
    from .vision import Perception

    return Perception(cfg.models.detector, cfg.models.classifier, cfg.models.imgsz, cfg.models.conf)


def cmd_run(args) -> int:
    from .pipeline import HivePipeline

    cfg = _cfg(args)
    source = args.source if args.source is not None else cfg.camera.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    live = isinstance(source, int) or str(source).startswith(("rtsp://", "http://", "https://"))
    storage = Storage(cfg.storage.data_dir)
    notifier = Notifier(cfg.alerts, dry_run=args.dry_run)
    pipe = HivePipeline(cfg, storage, notifier, _perception(cfg))
    dash_dir = Path(cfg.storage.data_dir) / "dashboard"

    def refresh() -> None:
        st = compute_day(storage, cfg, date.today().isoformat())
        if datetime.now().hour >= cfg.alerts.daily_summary_hour:
            notify_status(notifier, cfg, st)
        dashboard.render(storage, cfg, dash_dir)

    annotated = None
    if cfg.storage.save_annotated_video or args.annotate:
        annotated = Path(cfg.storage.data_dir) / "videos" / f"{datetime.now():%Y%m%d_%H%M%S}.mp4"
    print(f"Watching {source!r} ({'live' if live else 'file'}). Ctrl+C to stop.")
    try:
        summary = pipe.run(
            source, annotated_path=annotated, live=live, on_refresh=refresh, max_seconds=args.max_seconds
        )
    except KeyboardInterrupt:
        print("Stopped.")
        return 0
    for d in sorted(summary.days):
        compute_day(storage, cfg, d)
    dashboard.render(storage, cfg, dash_dir)
    print(json.dumps({k: v for k, v in summary.__dict__.items() if k != "days"}, default=str, indent=1))
    return 0


# -- demo ------------------------------------------------------------------------
def cmd_demo(args) -> int:
    from .pipeline import HivePipeline
    from .videoio import to_h264

    samples = Path(args.samples)
    manifest = yaml.safe_load((samples / "manifest.yaml").read_text(encoding="utf-8"))
    base = _cfg(args)
    base.language = args.lang or base.language
    out = Path(args.out)
    data_dir = out / "hive_data"
    storage = Storage(data_dir)
    base.storage.data_dir = str(data_dir)
    notifier = Notifier(base.alerts, dry_run=True)  # the demo never sends anything

    # 1. A month of simulated history for a demo hive with an Asian hornet problem.
    demo_cfg = copy.deepcopy(base)
    demo_cfg.hive.id, demo_cfg.hive.name = "demo", ("Ruche démo" if base.language == "fr" else "Demo hive")
    write_simulation(storage, demo_cfg, args.scenario, 30, seed=7, hive_id="demo")

    # 2. The real and synthetic sample clips through the full vision pipeline.
    clips_cfg = copy.deepcopy(base)
    clips_cfg.hive.id = "samples"
    clips_cfg.hive.name = "Échantillons vidéo" if base.language == "fr" else "Sample clips"
    storage.clear_hive("samples")
    perception = _perception(clips_cfg)
    rows = []
    start = datetime.combine(date.today(), datetime.min.time()) + timedelta(hours=10)
    clips = manifest["clips"]
    if args.only:
        clips = [c for c in clips if any(o in c["file"] for o in args.only)]
    for i, clip in enumerate(clips):
        cfg = copy.deepcopy(clips_cfg)
        cfg.entrance = EntranceCfg(**{**EntranceCfg().__dict__, **clip.get("entrance", {})})
        cfg.camera.process_fps = clip.get("process_fps", args.fps)
        pipe = HivePipeline(cfg, storage, notifier, perception)
        src = samples / clip["file"]
        dst = out / "videos" / (Path(clip["file"]).stem + "_annotated.mp4")
        print(f"[{i + 1}/{len(clips)}] {clip['file']} ...", flush=True)
        s = pipe.run(str(src), start_time=start + timedelta(minutes=2 * i), annotated_path=dst)
        playable = to_h264(dst, dst.with_name(dst.stem + "_web.mp4"))
        if playable != dst:
            dst.unlink(missing_ok=True)
        rows.append({"clip": clip, "summary": s, "video": playable.relative_to(out).as_posix()})
        print(
            f"    in={s.bees_in} out={s.bees_out} pollen={s.pollen_in} hornet_events={len(s.hornet_events)} "
            f"({s.fps:.1f} fps analysed)",
            flush=True,
        )
    compute_day(storage, clips_cfg, date.today().isoformat(), "samples")
    page = dashboard.render(storage, demo_cfg, out)
    _append_clips(page, rows, base.language)
    print(f"\nDashboard: {page.resolve()}")
    if not args.no_open:
        webbrowser.open(page.resolve().as_uri())
    return 0


def _append_clips(page: Path, rows: list[dict], lang: str) -> None:
    fr = lang == "fr"
    cards = []
    for r in rows:
        c, s = r["clip"], r["summary"]
        hornets = ", ".join(sorted({e["species"] for e in s.hornet_events})) or "-"
        cards.append(
            f'<div class="card"><h3>{html.escape(c.get("title", c["file"]))}</h3>'
            f'<video src="{html.escape(r["video"])}" controls muted loop playsinline preload="metadata" '
            f'style="width:100%;border-radius:8px"></video>'
            f'<div class="sub">{html.escape(c.get("note", ""))}</div>'
            f"<table><tr><th>{'entrées' if fr else 'in'}</th><th>{'sorties' if fr else 'out'}</th>"
            f"<th>pollen</th><th>{'frelons' if fr else 'hornets'}</th></tr>"
            f"<tr><td>{s.bees_in}</td><td>{s.bees_out}</td><td>{s.pollen_in}</td>"
            f"<td>{len(s.hornet_events)} ({html.escape(hornets)})</td></tr></table></div>"
        )
    title = "Échantillons analysés" if fr else "Analysed sample clips"
    block = f'<h2>{title}</h2><div class="grid">{"".join(cards)}</div>'
    text = page.read_text(encoding="utf-8").replace("<footer>", block + "<footer>", 1)
    page.write_text(text, encoding="utf-8")


# -- calibrate / test-alert / export --------------------------------------------
def cmd_calibrate(args) -> int:
    """Draw the configured zones on one frame so the user can check placement."""
    import cv2

    from . import annotate
    from .geometry import scale_points

    cfg = _cfg(args)
    source = args.source if args.source is not None else cfg.camera.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    ok, frame = False, None
    for _ in range(10):  # cameras need a few frames to settle exposure
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print(f"Cannot read from {source!r}")
        return 1
    h, w = frame.shape[:2]
    e = cfg.entrance
    if e.mode == "zone":
        annotate.draw_polygon(frame, scale_points(e.zone, w, h), annotate.ZONE, 3)
    else:
        a, b = scale_points(e.line, w, h)
        annotate.draw_line(frame, a, b, annotate.ZONE, 3)
    if e.hawk_zone:
        annotate.draw_polygon(frame, scale_points(e.hawk_zone, w, h), annotate.HAWK, 2, alpha=0)
    for i in range(1, 10):  # 10 % grid helps reading normalised coordinates
        cv2.line(frame, (w * i // 10, 0), (w * i // 10, h), (200, 200, 200), 1)
        cv2.line(frame, (0, h * i // 10), (w, h * i // 10), (200, 200, 200), 1)
        cv2.putText(frame, f"{i / 10:.1f}", (w * i // 10 + 3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{i / 10:.1f}", (3, h * i // 10 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(args.out, frame)
    print(f"Wrote {args.out}: green = entrance zone, red = hawking zone. Edit config.yaml and re-run.")
    return 0


def cmd_test_alert(args) -> int:
    cfg = _cfg(args)
    n = Notifier(cfg.alerts, dry_run=args.dry_run)
    n.send(Alert("Smart Hive", "Test notification / notification de test", tags=["bee"]))
    print("Channels:", ", ".join(n.channels()))
    return 0


def cmd_export(args) -> int:
    from ultralytics import YOLO

    cfg = _cfg(args)
    for path, imgsz in ((cfg.models.detector, cfg.models.imgsz), (cfg.models.classifier, 96)):
        if Path(path).exists():
            print(YOLO(path).export(format=args.format, imgsz=imgsz, half=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="smart-hive", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="analyse the bundled samples and open the dashboard")
    d.add_argument("--samples", default="samples")
    d.add_argument("--out", default="out/demo")
    d.add_argument("--config")
    d.add_argument("--lang", choices=["fr", "en"])
    d.add_argument("--fps", type=float, default=12.5)
    d.add_argument("--scenario", default="hornet", choices=SCENARIOS)
    d.add_argument("--only", nargs="*", help="substring filter on clip file names")
    d.add_argument("--no-open", action="store_true")
    d.set_defaults(func=cmd_demo)

    r = sub.add_parser("run", help="watch the camera (or a video file)")
    r.add_argument("--config")
    r.add_argument("--source")
    r.add_argument("--annotate", action="store_true", help="save an annotated video")
    r.add_argument("--max-seconds", type=float)
    r.add_argument("--dry-run", action="store_true", help="print alerts instead of sending them")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("simulate", help="write synthetic history to test alerts and the dashboard")
    s.add_argument("--config")
    s.add_argument("--scenario", default="healthy", choices=SCENARIOS)
    s.add_argument("--days", type=int, default=30)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--data-dir")
    s.set_defaults(func=cmd_simulate)

    st = sub.add_parser("status", help="compute and print today's status")
    st.add_argument("--config")
    st.add_argument("--notify", action="store_true")
    st.add_argument("--dry-run", action="store_true")
    st.set_defaults(func=cmd_status)

    db = sub.add_parser("dashboard", help="regenerate the dashboard page")
    db.add_argument("--config")
    db.add_argument("--out")
    db.set_defaults(func=cmd_dashboard)

    sv = sub.add_parser("serve", help="serve the dashboard on the local network")
    sv.add_argument("--config")
    sv.add_argument("--dir")
    sv.add_argument("--host")
    sv.add_argument("--port", type=int)
    sv.set_defaults(func=cmd_serve)

    c = sub.add_parser("calibrate", help="draw the configured zones on a camera frame")
    c.add_argument("--config")
    c.add_argument("--source")
    c.add_argument("--out", default="calibration.jpg")
    c.set_defaults(func=cmd_calibrate)

    ta = sub.add_parser("test-alert", help="send a test notification on the configured channels")
    ta.add_argument("--config")
    ta.add_argument("--dry-run", action="store_true")
    ta.set_defaults(func=cmd_test_alert)

    ex = sub.add_parser("export", help="export the models for faster CPU inference")
    ex.add_argument("--config")
    ex.add_argument("--format", default="openvino", choices=["onnx", "openvino", "ncnn"])
    ex.set_defaults(func=cmd_export)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
