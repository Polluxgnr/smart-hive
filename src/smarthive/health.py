"""Daily colony-health status judged against the hive's own history.

Each signal is reduced to one value per day. The baseline is the median of the
last ``window_days`` *normal* days and the spread is the scaled MAD, so one
odd day does not move the baseline and a weak colony is not flagged every day
just for being weak. See docs/design.md, section 5.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

Level = Literal["learning", "ok", "watch", "check"]
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class SignalSpec:
    name: str
    bad: Literal["up", "down"]
    floor_abs: float = 0.0  # minimum spread, absolute units
    floor_rel: float = 0.0  # minimum spread, fraction of the baseline median


SIGNALS: dict[str, SignalSpec] = {
    s.name: s
    for s in (
        SignalSpec("activity", "down", floor_rel=0.12),
        SignalSpec("balance", "up", floor_abs=0.03),
        SignalSpec("pollen_share", "down", floor_abs=0.02, floor_rel=0.15),
        SignalSpec("crowding", "up", floor_abs=0.5, floor_rel=0.15),
        SignalSpec("agitation", "up", floor_abs=0.02, floor_rel=0.15),
    )
}


@dataclass(frozen=True)
class HealthConfig:
    window_days: int = 21
    min_days: int = 5
    watch_z: float = 2.5
    check_z: float = 3.5
    consecutive_check_days: int = 2
    daylight: tuple[int, int] = (8, 20)  # local hours [start, end)
    min_traffic_for_ratios: int = 100
    hornet_check_per_hour: float = 3.0
    cold_max_temp_c: float = 12.0
    rainy_mm: float = 2.0


@dataclass
class HourRow:
    """One hour of aggregates for one hive (what the pipeline stores)."""

    hour: int  # 0-23 local time
    observed_s: float = 3600.0
    bees_in: int = 0
    bees_out: int = 0
    pollen_in: int = 0
    tracks: int = 0
    erratic_tracks: int = 0
    stationary_bee_s: float = 0.0
    velutina_events: int = 0
    other_vespid_events: int = 0


def daily_values(rows: Iterable[HourRow], cfg: HealthConfig | None = None) -> dict[str, float | None]:
    """Reduce one day of hourly rows to the signal values."""
    cfg = cfg or HealthConfig()
    start, end = cfg.daylight
    day = [r for r in rows if start <= r.hour < end and r.observed_s > 0]
    out: dict[str, float | None] = dict.fromkeys(SIGNALS)
    out.update(traffic=0.0, velutina_events=0.0, velutina_peak_hour=0.0, other_vespid_events=0.0)
    if not day:
        return out
    observed_h = sum(r.observed_s for r in day) / 3600.0
    b_in = sum(r.bees_in for r in day)
    b_out = sum(r.bees_out for r in day)
    traffic = b_in + b_out
    out["traffic"] = float(traffic)
    # Rate per observed hour so a camera that ran half the day is comparable.
    out["activity"] = traffic / observed_h if observed_h > 0 else None
    if traffic >= cfg.min_traffic_for_ratios:
        out["balance"] = (b_out - b_in) / traffic
    if b_in >= cfg.min_traffic_for_ratios // 2:
        out["pollen_share"] = sum(r.pollen_in for r in day) / b_in
    out["crowding"] = sum(r.stationary_bee_s for r in day) / (observed_h * 3600.0)
    tracks = sum(r.tracks for r in day)
    if tracks >= cfg.min_traffic_for_ratios:
        out["agitation"] = sum(r.erratic_tracks for r in day) / tracks
    out["velutina_events"] = float(sum(r.velutina_events for r in rows))
    out["other_vespid_events"] = float(sum(r.other_vespid_events for r in rows))
    out["velutina_peak_hour"] = float(max((r.velutina_events for r in rows), default=0))
    return out


def robust_baseline(values: Sequence[float], spec: SignalSpec) -> tuple[float, float]:
    """(median, robust sigma) with the per-signal floor applied."""
    med = statistics.median(values)
    mad = statistics.median(abs(v - med) for v in values)
    sigma = MAD_TO_SIGMA * mad
    floor = max(spec.floor_abs, spec.floor_rel * abs(med))
    return med, max(sigma, floor, 1e-9)


def badness(value: float, median: float, sigma: float, spec: SignalSpec) -> float:
    """Robust z-score signed so that positive means 'worse than usual'."""
    z = (value - median) / sigma
    return z if spec.bad == "up" else -z


@dataclass
class SignalResult:
    name: str
    value: float | None
    median: float | None = None
    sigma: float | None = None
    z: float | None = None
    level: Level = "learning"
    note: str | None = None  # e.g. "weather"


@dataclass
class DayStatus:
    day: str
    level: Level
    open_hive: bool
    signals: dict[str, SignalResult]
    hornet_level: str = "none"
    reasons: list[str] = field(default_factory=list)  # i18n keys, see i18n.py
    anomalous: bool = False  # excluded from future baselines


@dataclass
class DayRecord:
    """What is stored per day and fed back as history."""

    day: str
    values: Mapping[str, float | None]
    levels: Mapping[str, str] = field(default_factory=dict)
    anomalous: bool = False


def _level(z: float, cfg: HealthConfig) -> Level:
    if z >= cfg.check_z:
        return "check"
    if z >= cfg.watch_z:
        return "watch"
    return "ok"


_RANK = {"learning": 0, "ok": 1, "watch": 2, "check": 3}


def evaluate_day(
    day: str,
    values: Mapping[str, float | None],
    history: Sequence[DayRecord],
    cfg: HealthConfig | None = None,
    weather: Mapping[str, float] | None = None,
) -> DayStatus:
    """Judge one day against the previous days of the same hive.

    ``history`` is ordered oldest -> newest and must not contain ``day``.
    ``weather`` may hold ``max_temp_c`` and ``rain_mm`` for the day.
    """
    cfg = cfg or HealthConfig()
    usable = [h for h in history if not h.anomalous][-cfg.window_days :]
    bad_weather = bool(
        weather and (weather.get("max_temp_c", 99) < cfg.cold_max_temp_c or weather.get("rain_mm", 0) >= cfg.rainy_mm)
    )
    results: dict[str, SignalResult] = {}
    reasons: list[str] = []
    open_hive = False
    for name, spec in SIGNALS.items():
        value = values.get(name)
        res = SignalResult(name, value)
        past = [h.values[name] for h in usable if h.values.get(name) is not None]
        if value is not None and len(past) >= cfg.min_days:
            med, sigma = robust_baseline(past, spec)  # type: ignore[arg-type]
            z = badness(value, med, sigma, spec)
            res.median, res.sigma, res.z, res.level = med, sigma, z, _level(z, cfg)
            if name == "activity" and res.level != "ok" and bad_weather:
                res.level, res.note = "ok", "weather"
            if res.level in ("watch", "check"):
                reasons.append(f"{name}_{res.level}")
            need = cfg.consecutive_check_days - 1
            recent = history[-need:] if need > 0 else []
            if res.level == "check" and len(recent) == need and all(h.levels.get(name) == "check" for h in recent):
                open_hive = True
        results[name] = res

    v_peak = float(values.get("velutina_peak_hour") or 0)
    if v_peak <= 0:
        hornet_level = "none"
    elif v_peak < cfg.hornet_check_per_hour:
        hornet_level = "present"
        reasons.append("hornet_present")
    else:
        hornet_level = "check"
        reasons.append("hornet_check")

    known = [r.level for r in results.values() if r.level != "learning"]
    if not known:
        level: Level = "learning"
    else:
        level = max(known, key=lambda lv: _RANK[lv])  # type: ignore[assignment]
    if hornet_level == "check":
        level = "check"
    elif hornet_level == "present" and level in ("ok", "learning"):
        level = "watch" if known else level
    if open_hive:
        reasons.insert(0, "open_hive")
    if bad_weather and any(r.note == "weather" for r in results.values()):
        reasons.append("quiet_weather")
    anomalous = any(r.level == "check" for r in results.values()) or bad_weather
    return DayStatus(day, level, open_hive, results, hornet_level, reasons, anomalous)


def to_record(status: DayStatus, values: Mapping[str, float | None]) -> DayRecord:
    return DayRecord(
        status.day,
        dict(values),
        {k: r.level for k, r in status.signals.items()},
        status.anomalous,
    )
