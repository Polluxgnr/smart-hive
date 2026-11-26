import pytest

from smarthive.alerts import Alert, Notifier
from smarthive.config import AlertsCfg, Config, load_config, validate
from smarthive.hornet import HornetMonitor, pressure_level
from smarthive.tracks import SPECIES, Track


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_console_only_by_default(capsys):
    n = Notifier(AlertsCfg())
    assert n.channels() == ["console"]
    n.send(Alert("Title", "Body"))
    assert "[ALERT] Title: Body" in capsys.readouterr().out


def test_cooldown_suppresses_repeats():
    clock = FakeClock()
    n = Notifier(AlertsCfg(), clock=clock)
    assert n.send(Alert("a", "b", key="hornet:h1"), cooldown_s=900)
    clock.now = 100
    assert not n.send(Alert("a", "b", key="hornet:h1"), cooldown_s=900)
    assert n.send(Alert("a", "b", key="hornet:h2"), cooldown_s=900)  # other hive
    clock.now = 1000
    assert n.send(Alert("a", "b", key="hornet:h1"), cooldown_s=900)


def test_dry_run_never_touches_the_network(monkeypatch, capsys):
    cfg = AlertsCfg()
    cfg.ntfy.enabled, cfg.ntfy.topic = True, "my-hive-topic"
    cfg.email.enabled, cfg.email.smtp_host, cfg.email.to = True, "smtp.example.org", ["me@example.org"]

    def boom(*a, **k):
        raise AssertionError("network used in dry-run")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    monkeypatch.setattr("smtplib.SMTP", boom)
    n = Notifier(cfg, dry_run=True)
    n.send(Alert("Hornet", "Asian hornet hovering"))
    out = capsys.readouterr().out
    assert "[dry-run] ntfy POST https://ntfy.sh/my-hive-topic" in out
    assert "[dry-run] e-mail" in out
    assert [c for c, _ in n.sent] == ["console", "ntfy", "email"]


def test_broken_channel_does_not_raise(monkeypatch):
    cfg = AlertsCfg()
    cfg.ntfy.enabled, cfg.ntfy.topic = True, "t"

    def fail(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    n = Notifier(cfg)
    assert n.send(Alert("x", "y"))
    assert [c for c, _ in n.sent] == ["console"]


def test_email_password_comes_from_environment_only(monkeypatch):
    cfg = AlertsCfg()
    cfg.email.enabled, cfg.email.smtp_host, cfg.email.to, cfg.email.username = True, "h", ["a@b"], "user"
    sent = {}

    class FakeSMTP:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self, **k):
            pass

        def login(self, user, password):
            sent["login"] = (user, password)

        def send_message(self, msg):
            sent["subject"] = msg["Subject"]

    monkeypatch.setattr("smtplib.SMTP", FakeSMTP)
    monkeypatch.setenv("SMART_HIVE_SMTP_PASSWORD", "s3cret")
    Notifier(cfg).send(Alert("Subject", "Body"))
    assert sent == {"login": ("user", "s3cret"), "subject": "Subject"}


def test_config_rejects_ntfy_without_topic(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("alerts:\n  ntfy:\n    enabled: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="topic"):
        load_config(p)


def test_config_rejects_unknown_keys_and_pixel_coordinates(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("hive:\n  nmae: typo\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_config(p)
    cfg = Config()
    cfg.entrance.zone = [[100, 200], [300, 200], [300, 400]]
    with pytest.raises(ValueError, match="normalised"):
        validate(cfg)


def test_example_config_is_valid():
    cfg = load_config("config.example.yaml")
    assert cfg.alerts.ntfy.enabled is False and cfg.alerts.email.enabled is False


# -- hornet module -----------------------------------------------------------------
DT = 0.1


def probs_for(species, p=0.9):
    probs = [0.0] * len(SPECIES)
    probs[SPECIES.index(species)] = p
    probs[0] += 1 - p
    return probs


def feed(mon, points, species="vespa_velutina", detector_vespid=True, p=0.9, t0=0.0, track_id=1, bees=None):
    """Feed one track frame by frame. Returns (opened, closed) collected over all frames."""
    tr = Track(track_id, first_t=t0)
    opened, closed = [], []
    for i, pt in enumerate(points):
        t = t0 + i * DT
        tr.add(t, pt, 30.0, detector_vespid, 0.8)
        tr.add_probs(probs_for(species, p), i)
        o, c = mon.update(t, {track_id: tr}, {track_id}, bees(pt) if bees else ())
        opened += o
        closed += c
    return opened, closed, tr


def hover(n, cx=500, cy=300, r=20):
    return [(cx + (r if i % 4 < 2 else -r), cy + (r if i % 2 else -r)) for i in range(n)]


def test_hovering_velutina_opens_one_event_and_closes_it():
    mon = HornetMonitor(hawk_zone=None, frame_diag=1000, min_seconds=3.0)
    opened, _, _ = feed(mon, hover(20))
    assert opened == []  # 2 s of hovering is not enough
    opened, _, _ = feed(mon, hover(30), t0=2.0)
    assert len(opened) == 1 and opened[0].is_velutina
    _, closed = mon.update(4.95, {}, set())
    assert closed == []  # short gaps do not end the event
    _, closed = mon.update(7.5, {}, set())
    assert len(closed) == 1 and closed[0].duration == pytest.approx(4.9, abs=0.2)


def test_fragmented_tracks_still_make_one_event():
    """Fast flight breaks tracks; presence over the window is what counts."""
    mon = HornetMonitor(hawk_zone=None, frame_diag=1000)
    opened = []
    for k in range(5):  # five 1-second track pieces with new ids
        o, _, _ = feed(mon, hover(10), t0=k * 1.0, track_id=100 + k)
        opened += o
    assert len(opened) == 1


def test_hornet_flying_through_is_not_hawking():
    mon = HornetMonitor(hawk_zone=None, frame_diag=1000, min_seconds=3.0)
    opened, _, _ = feed(mon, [(i * 25.0, 300) for i in range(40)])  # crosses 1000 px in 4 s
    assert opened == []


def test_hovering_bee_is_not_a_hornet_event():
    mon = HornetMonitor(hawk_zone=None, frame_diag=1000)
    assert feed(mon, hover(40), species="bee", detector_vespid=False)[0] == []


def test_sitting_vespid_is_not_hawking():
    mon = HornetMonitor(hawk_zone=None, frame_diag=1000)
    # 4 s without moving: a cluster at the slot, not a hornet
    assert feed(mon, [(500.0, 300.0)] * 40)[0] == []


def test_classifier_alone_needs_to_be_sure():
    mon = HornetMonitor(hawk_zone=None, frame_diag=1000)
    assert feed(mon, hover(40), detector_vespid=False, p=0.7)[0] == []  # unsure, detector says bee
    mon = HornetMonitor(hawk_zone=None, frame_diag=1000)
    assert len(feed(mon, hover(40), detector_vespid=False, p=0.9)[0]) == 1  # very sure


def test_hawking_outside_zone_is_ignored():
    zone = [(0, 0), (200, 0), (200, 200), (0, 200)]
    mon = HornetMonitor(hawk_zone=zone, frame_diag=1000)
    assert feed(mon, hover(40))[0] == []
    mon2 = HornetMonitor(hawk_zone=zone, frame_diag=1000)
    assert len(feed(mon2, hover(40, cx=100, cy=100))[0]) == 1


def test_strikes_count_close_approaches_once_per_bee():
    mon = HornetMonitor(hawk_zone=None, frame_diag=1000)
    near = lambda pt: [(7, (pt[0] + 10, pt[1])), (8, (pt[0] + 500, pt[1]))]  # noqa: E731
    opened, _, _ = feed(mon, hover(50), bees=near)
    assert len(opened) == 1
    assert mon.open_event().strikes == 1


def test_species_comes_from_the_vote():
    mon = HornetMonitor(hawk_zone=None, frame_diag=1000)
    opened, _, _ = feed(mon, hover(40), species="vespa_crabro")
    assert opened[0].species == "vespa_crabro" and not opened[0].is_velutina


def test_track_vote_beats_single_bad_frame():
    tr = Track(1, first_t=0)
    bee = [0.9, 0.05, 0.03, 0.01, 0.01]
    hornet = [0.05, 0.0, 0.9, 0.03, 0.02]
    for i in range(9):
        tr.add_probs(bee, i)
    tr.add_probs(hornet, 9)
    assert tr.species == "bee"


def test_pressure_levels():
    assert pressure_level(0) == "none"
    assert pressure_level(1) == "present"
    assert pressure_level(5) == "check"
