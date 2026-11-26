"""The monitoring loop: frames -> tracks -> counts, behaviour, hornet events -> storage and alerts."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from . import annotate
from .alerts import Alert, Notifier
from .capture import open_source
from .config import Config
from .counting import EntranceCounter, LineRule, ZoneRule
from .geometry import scale_points
from .health import HourRow
from .hornet import HornetEvent, HornetMonitor
from .i18n import t
from .storage import Storage
from .tracks import VESPIDS, Track
from .vision import Detection, Perception

log = logging.getLogger(__name__)


@dataclass
class RunSummary:
    frames: int = 0
    analysed: int = 0
    seconds: float = 0.0
    bees_in: int = 0
    bees_out: int = 0
    pollen_in: int = 0
    hornet_events: list[dict] = field(default_factory=list)
    alerts: int = 0
    days: set[str] = field(default_factory=set)
    fps: float = 0.0
    annotated_video: str | None = None


class _HourBucket:
    def __init__(self, start: datetime):
        self.start = start.replace(minute=0, second=0, microsecond=0)
        self.row = HourRow(hour=self.start.hour, observed_s=0.0)

    @property
    def day(self) -> str:
        return self.start.date().isoformat()


class HivePipeline:
    def __init__(
        self,
        cfg: Config,
        storage: Storage,
        notifier: Notifier,
        perception: Perception,
        simulated: bool = False,
    ):
        self.cfg, self.storage, self.notifier, self.perception = cfg, storage, notifier, perception
        self.simulated = simulated
        self.lang = cfg.language

    # -- geometry ------------------------------------------------------------
    def _setup(self, width: int, height: int) -> None:
        e = self.cfg.entrance
        self.zone = scale_points(e.zone, width, height)
        self.line = scale_points(e.line, width, height)
        self.hawk_zone = scale_points(e.hawk_zone, width, height) if e.hawk_zone else None
        if e.mode == "zone":
            rule: ZoneRule | LineRule = ZoneRule(self.zone)
        else:
            inside = scale_points([e.inside_point], width, height)[0]
            rule = LineRule(self.line[0], self.line[1], inside, e.margin * height)
        self.counter = EntranceCounter(rule, max_age=self._max_age, min_travel=e.min_travel * width)
        self.hornets = HornetMonitor(
            self.hawk_zone,
            math.hypot(width, height),
            min_seconds=self.cfg.hornet.hawk_min_seconds,
            max_spread=self.cfg.hornet.max_spread,
        )

    # -- main loop -----------------------------------------------------------
    def run(
        self,
        source,
        start_time: datetime | None = None,
        max_seconds: float | None = None,
        annotated_path: str | Path | None = None,
        live: bool = False,
        on_refresh=None,
    ) -> RunSummary:
        cap = open_source(source, self.cfg.camera.width, self.cfg.camera.height, self.cfg.camera.process_fps)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video source {source!r}")
        if live:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera.height)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if src_fps <= 1 or src_fps > 240:
            src_fps = 25.0
        proc_fps = min(self.cfg.camera.process_fps, src_fps)
        stride = max(1, round(src_fps / proc_fps))
        dt = stride / src_fps
        self._max_age = max(1.5, 30 * dt)  # match the tracker's buffer (30 analysed frames)
        start_time = start_time or datetime.now()
        summary = RunSummary()
        writer = None
        tracks: dict[int, Track] = {}
        bucket: _HourBucket | None = None
        wall0 = time.perf_counter()
        last_refresh = time.monotonic()
        frame_idx = -1
        self.perception.reset()
        geometry_ready = False
        t_video = 0.0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                summary.frames += 1
                if live:
                    # Camera: analyse at most process_fps frames per second, drop the rest.
                    t_now = time.perf_counter() - wall0
                    if summary.analysed and t_now - t_video < 1 / proc_fps:
                        continue
                    dt = t_now - t_video if summary.analysed else 1 / proc_fps
                    t_video = t_now
                else:
                    if frame_idx % stride:
                        continue
                    t_video = frame_idx / src_fps
                if max_seconds is not None and t_video > max_seconds:
                    break
                if not geometry_ready:
                    self._setup(frame.shape[1], frame.shape[0])
                    geometry_ready = True
                now = start_time + timedelta(seconds=t_video)
                if bucket is None or now >= bucket.start + timedelta(hours=1):
                    if bucket is not None:
                        self._flush(bucket, summary)
                    bucket = _HourBucket(now)
                bucket.row.observed_s += dt
                summary.analysed += 1

                dets = self.perception.track(frame)
                self._update_tracks(tracks, dets, t_video, summary.analysed, frame)
                visible = {d.track_id for d in dets}

                # Counting: bee tracks only (a vespid is never a forager).
                obs = {d.track_id: d.center for d in dets if tracks[d.track_id].vespid_score < 0.5}
                for c in self.counter.update(t_video, obs):
                    self._count(c.track_id, c.direction, tracks, bucket, summary)

                # Behaviour: stationary bees (bearding / fanning proxy).
                n_stationary = sum(1 for i in visible if tracks[i].vespid_score < 0.5 and tracks[i].is_stationary(20.0))
                bucket.row.stationary_bee_s += n_stationary * dt

                # Finished tracks: erratic-motion statistics, then forget them.
                for tid in [i for i, tr in tracks.items() if t_video - tr.last_t > self._max_age]:
                    self._retire(tracks.pop(tid), bucket)

                bee_pts = [(i, tracks[i].points[-1]) for i in visible if tracks[i].vespid_score < 0.5]
                opened, closed = self.hornets.update(t_video, tracks, visible, bee_pts)
                for ev in opened:
                    self._on_hornet_open(ev, frame, tracks, now, summary)
                for ev in closed:
                    self._on_hornet_close(ev, bucket, start_time, summary)

                if annotated_path is not None:
                    vis = self._draw(frame, dets, tracks, summary)
                    if writer is None:
                        Path(annotated_path).parent.mkdir(parents=True, exist_ok=True)
                        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(annotated_path), fourcc, 1 / dt, vis.shape[1::-1])
                    writer.write(vis)

                if live and on_refresh and time.monotonic() - last_refresh > 60 * self.cfg.dashboard.refresh_minutes:
                    self._flush(bucket, summary)
                    bucket = _HourBucket(now)
                    on_refresh()
                    last_refresh = time.monotonic()
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        if geometry_ready:
            for c in self.counter.flush():
                self._count(c.track_id, c.direction, tracks, bucket, summary)
            for tr in tracks.values():
                self._retire(tr, bucket)
            for ev in self.hornets.close_all(t_video):
                self._on_hornet_close(ev, bucket, start_time, summary)
        if bucket is not None:
            self._flush(bucket, summary)
        summary.seconds = t_video
        summary.fps = summary.analysed / max(1e-6, time.perf_counter() - wall0)
        summary.annotated_video = str(annotated_path) if annotated_path else None
        return summary

    # -- helpers -------------------------------------------------------------
    def _update_tracks(self, tracks, dets: list[Detection], t: float, frame_no: int, frame: np.ndarray) -> None:
        for d in dets:
            tr = tracks.get(d.track_id)
            if tr is None:
                tr = tracks[d.track_id] = Track(d.track_id, first_t=t)
            tr.add(t, d.center, d.size, d.label == "vespid", d.conf)
        every = max(1, self.cfg.models.classify_every)
        need = [d for d in dets if frame_no - tracks[d.track_id].last_classified_frame >= every]
        if need:
            probs = self.perception.classify(frame, [d.box for d in need])
            for d, p in zip(need, probs, strict=False):
                tracks[d.track_id].add_probs(p, frame_no)

    def _count(self, tid: int, direction: str, tracks, bucket: _HourBucket | None, summary: RunSummary) -> None:
        if bucket is None:
            return
        tr = tracks.get(tid)
        if direction == "in":
            bucket.row.bees_in += 1
            summary.bees_in += 1
            if tr is not None and tr.carries_pollen:
                bucket.row.pollen_in += 1
                summary.pollen_in += 1
        else:
            bucket.row.bees_out += 1
            summary.bees_out += 1

    def _retire(self, tr: Track, bucket: _HourBucket | None) -> None:
        if bucket is None or tr.vespid_score >= 0.5 or len(tr.points) < 5:
            return
        bucket.row.tracks += 1
        if tr.is_erratic():
            bucket.row.erratic_tracks += 1

    def _flush(self, bucket: _HourBucket, summary: RunSummary) -> None:
        self.storage.add_hour(self.cfg.hive.id, bucket.day, bucket.row, simulated=self.simulated)
        summary.days.add(bucket.day)
        bucket.row = HourRow(hour=bucket.row.hour, observed_s=0.0)

    def _on_hornet_open(self, ev: HornetEvent, frame, tracks, now: datetime, summary: RunSummary) -> None:
        tr = tracks.get(ev.track_id)
        vis = frame.copy()
        box = None
        if tr is not None and tr.points:
            x, y = tr.points[-1]
            s = tr.body_length * 1.2
            box = (x - s / 2, y - s / 2, x + s / 2, y + s / 2)
            label = f"{annotate.species_label(ev.species, self.lang)} {ev.confidence:.0%}"
            annotate.draw_box(vis, box, label, annotate.COLOURS.get(ev.species, annotate.HAWK), 3)
        name = f"{now:%Y%m%d_%H%M%S}_{self.cfg.hive.id}_{ev.event_id}.jpg"
        path = self.storage.data_dir / "snapshots" / name
        cv2.imwrite(str(path), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        ev.snapshot_jpeg = name.encode()
        ev.box = box
        ev._ts = now  # type: ignore[attr-defined]
        if ev.species in self.cfg.hornet.alert_species:
            sent = self.notifier.send(
                Alert(
                    title=t("alert.hornet.title", self.lang, hive=self.cfg.hive.name),
                    body=t(
                        "alert.hornet.body",
                        self.lang,
                        species=t(f"species.{ev.species}", self.lang),
                        duration=self.cfg.hornet.hawk_min_seconds,
                        conf=ev.confidence,
                    ),
                    priority=4,
                    tags=["warning", "bee"],
                    attachment=path,
                    key=f"hornet:{self.cfg.hive.id}",
                ),
                cooldown_s=60 * self.cfg.hornet.cooldown_minutes,
            )
            ev._alerted = sent  # type: ignore[attr-defined]
            summary.alerts += int(sent)

    def _on_hornet_close(self, ev: HornetEvent, bucket, start_time: datetime, summary: RunSummary) -> None:
        ts: datetime = getattr(ev, "_ts", start_time + timedelta(seconds=ev.start_t))
        snapshot = ev.snapshot_jpeg.decode() if ev.snapshot_jpeg else None
        self.storage.add_event(
            self.cfg.hive.id,
            ts.isoformat(timespec="seconds"),
            ev.species,
            ev.confidence,
            ev.duration,
            ev.strikes,
            f"snapshots/{snapshot}" if snapshot else None,
            alerted=getattr(ev, "_alerted", False),
            simulated=self.simulated,
        )
        if bucket is not None:
            if ev.is_velutina:
                bucket.row.velutina_events += 1
            else:
                bucket.row.other_vespid_events += 1
        summary.hornet_events.append(
            {
                "species": ev.species,
                "confidence": round(ev.confidence, 3),
                "duration_s": round(ev.duration, 1),
                "strikes": ev.strikes,
                "snapshot": snapshot,
            }
        )

    def _draw(self, frame, dets: list[Detection], tracks, summary: RunSummary) -> np.ndarray:
        vis = frame.copy()
        if self.cfg.entrance.mode == "zone":
            annotate.draw_polygon(vis, self.zone, annotate.ZONE)
        else:
            annotate.draw_line(vis, self.line[0], self.line[1])
        if self.hawk_zone:
            annotate.draw_polygon(vis, self.hawk_zone, annotate.HAWK, 1, alpha=0.0)
        warn = None
        for d in dets:
            tr = tracks[d.track_id]
            sp = tr.species
            colour = annotate.COLOURS.get(sp, annotate.COLOURS["bee"])
            label = ""
            if sp in VESPIDS or sp == "vespid":
                label = f"{annotate.species_label(sp, self.lang)} {tr.species_confidence:.0%}"
                if self.hornets.open_event(d.track_id) is not None:
                    action = "VOL STATIONNAIRE" if self.lang == "fr" else "HAWKING"
                    warn = f"{annotate.species_label(sp, self.lang).upper()} - {action}"
            elif sp == "bee_pollen":
                label = "pollen"
            annotate.draw_box(vis, d.box, label, colour, 3 if label else 2)
        lang = self.lang
        lines = [
            f"{t('dash.in', lang)}: {summary.bees_in}   {t('dash.out', lang)}: {summary.bees_out}",
            f"{annotate.species_label('bee_pollen', lang)}: {summary.pollen_in}",
        ]
        if summary.hornet_events or warn:
            n = len(summary.hornet_events) + (1 if warn else 0)
            lines.append(f"{'Frelons' if lang == 'fr' else 'Hornets'}: {n}")
        annotate.draw_panel(vis, lines, warn)
        return vis
