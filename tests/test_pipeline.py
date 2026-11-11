"""End-to-end test of the monitoring loop with a scripted fake detector (no model weights needed)."""

from datetime import datetime

import cv2
import numpy as np

from smarthive import dashboard
from smarthive.alerts import Notifier
from smarthive.config import Config
from smarthive.pipeline import HivePipeline
from smarthive.status import compute_day
from smarthive.storage import Storage
from smarthive.tracks import SPECIES
from smarthive.vision import Detection

W, H, FPS, N = 320, 180, 10, 80


class FakePerception:
    """Bee 1 walks into the entrance slot, bee 2 walks out, track 3 is a hovering Asian hornet."""

    def __init__(self):
        self.i = -1

    def reset(self):
        self.i = -1

    def track(self, frame):
        self.i += 1
        i = self.i
        dets = []
        if i < 20:
            y = 150 - 6 * i  # from y=150 up to the slot band at the top
            dets.append(Detection(1, (150, y - 6, 162, y + 6), "bee", 0.9))
        if 25 <= i < 45:
            y = 30 + 6 * (i - 25)
            dets.append(Detection(2, (100, y - 6, 112, y + 6), "bee", 0.9))
        if 10 <= i < 70:
            x = 220 + 8 * np.sin(i / 3)
            dets.append(Detection(3, (x - 12, 100, x + 12, 124), "vespid", 0.8))
        return dets

    def classify(self, frame, boxes):
        out = []
        for b in boxes:
            p = [0.0] * len(SPECIES)
            if b[2] - b[0] > 20:  # the big one is the hornet
                p[SPECIES.index("vespa_velutina")] = 0.9
                p[0] = 0.1
            else:
                p[0] = 1.0
            out.append(p)
        return out


def make_video(path):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (W, H))
    for _ in range(N):
        vw.write(np.full((H, W, 3), 120, np.uint8))
    vw.release()


def test_pipeline_counts_bees_and_raises_one_hornet_alert(tmp_path, capsys):
    video = tmp_path / "clip.mp4"
    make_video(video)
    cfg = Config()
    cfg.storage.data_dir = str(tmp_path / "data")
    cfg.camera.process_fps = FPS
    cfg.entrance.zone = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.25], [0.0, 0.25]]  # slot band: top quarter
    storage = Storage(cfg.storage.data_dir)
    notifier = Notifier(cfg.alerts)
    pipe = HivePipeline(cfg, storage, notifier, FakePerception())
    s = pipe.run(str(video), start_time=datetime(2026, 7, 1, 11, 0), annotated_path=tmp_path / "out.mp4")

    assert (s.bees_in, s.bees_out) == (1, 1)
    assert len(s.hornet_events) == 1 and s.hornet_events[0]["species"] == "vespa_velutina"
    assert s.alerts == 1
    assert "[ALERT]" in capsys.readouterr().out
    hours = storage.hours(cfg.hive.id, "2026-07-01")
    assert hours[0].bees_in == 1 and hours[0].velutina_events == 1
    ev = storage.events(cfg.hive.id)
    assert ev[0]["snapshot"] and (tmp_path / "data" / ev[0]["snapshot"]).exists()
    assert (tmp_path / "out.mp4").stat().st_size > 0

    status = compute_day(storage, cfg, "2026-07-01")
    assert status.hornet_level == "present"
    page = dashboard.render(storage, cfg, tmp_path / "dash")
    html = page.read_text(encoding="utf-8")
    assert cfg.hive.name in html and "snapshots/" in html
