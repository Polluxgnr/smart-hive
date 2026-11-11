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
def make_track(points, species="vespa_velutina", dt=0.1):
    tr = Track(1, first_t=0.0)
    probs = [0.0] * len(SPECIES)
    probs[SPECIES.index(species)] = 0.9
    probs[0] += 0.1
    for i, p in enumerate(points):
        tr.add(i * dt, p, 30.0, species != "bee", 0.8)
        tr.add_probs(probs, i)
    return tr


def hover(n, cx=500, cy=300, r=20):
    return [(cx + (r if i % 4 < 2 else -r), cy + (r if i % 2 else -r)) for i in range(n)]


def test_hovering_velutina_opens_one_event_and_closes_it():
    mon = HornetMonitor(hawk_zone=None, frame_diag=1000, min_seconds=3.0)
    tr = make_track(hover(10))
    opened, _ = mon.update(1.0, {1: tr}, {1})
    assert opened == []  # 1 s of hovering is not enough
    tr = make_track(hover(40))
    opened, _ = mon.update(4.0, {1: tr}, {1})
    assert len(opened) == 1 and opened[0].is_velutina
    opened, _ = mon.update(4.1, {1: tr}, {1})
    assert opened == []  # same track, same event
    _, closed = mon.update(6.0, {}, set())
    assert len(closed) == 1 and closed[0].duration == pytest.approx(2.0)


def test_hornet_flying_through_is_not_hawking():
    mon = HornetMonitor(hawk_zone=None, frame_diag=1000, min_seconds=3.0)
    tr = make_track([(i * 25.0, 300) for i in range(40)])  # crosses 1000 px in 4 s
    opened, _ = mon.update(4.0, {1: tr}, {1})
    assert opened == []


def test_hovering_bee_is_not_a_hornet_event():
    mon = HornetMonitor(hawk_zone=None, frame_diag=1000)
    tr = make_track(hover(40), species="bee")
    assert mon.update(4.0, {1: tr}, {1})[0] == []


def test_hawking_outside_zone_is_ignored():
    zone = [(0, 0), (200, 0), (200, 200), (0, 200)]
    mon = HornetMonitor(hawk_zone=zone, frame_diag=1000)
    assert mon.update(4.0, {1: make_track(hover(40))}, {1})[0] == []
    mon2 = HornetMonitor(hawk_zone=zone, frame_diag=1000)
    assert len(mon2.update(4.0, {1: make_track(hover(40, cx=100, cy=100))}, {1})[0]) == 1


def test_strikes_count_close_approaches_once_per_bee():
    mon = HornetMonitor(hawk_zone=None, frame_diag=1000)
    tr = make_track(hover(40))
    mon.update(4.0, {1: tr}, {1})
    hx, hy = tr.points[-1]
    mon.update(4.1, {1: tr}, {1}, bee_points=[(7, (hx + 10, hy)), (8, (hx + 500, hy))])
    mon.update(4.2, {1: tr}, {1}, bee_points=[(7, (hx + 5, hy))])
    assert mon.open_event(1).strikes == 1


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
