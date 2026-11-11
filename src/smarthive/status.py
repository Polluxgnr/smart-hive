"""Glue between storage, health logic, weather and alerts for the daily status."""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

from .alerts import Alert, Notifier
from .config import Config
from .health import DayStatus, daily_values, evaluate_day
from .i18n import t
from .storage import Storage

log = logging.getLogger(__name__)


def fetch_weather(lat: float, lon: float, day: str) -> dict[str, float] | None:
    """Daily max temperature and rain from Open-Meteo (free, no key). None on failure."""
    q = urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,precipitation_sum",
            "start_date": day,
            "end_date": day,
            "timezone": "auto",
        }
    )
    try:
        req = urllib.request.Request(
            f"https://api.open-meteo.com/v1/forecast?{q}", headers={"User-Agent": "smart-hive/0.1"}
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            daily = json.loads(r.read())["daily"]
        return {"max_temp_c": daily["temperature_2m_max"][0], "rain_mm": daily["precipitation_sum"][0]}
    except Exception as exc:
        log.warning("weather unavailable: %s", exc)
        return None


def compute_day(
    storage: Storage,
    cfg: Config,
    day: str,
    hive_id: str | None = None,
    weather: dict[str, float] | None = None,
) -> DayStatus:
    hive_id = hive_id or cfg.hive.id
    values = daily_values(storage.hours(hive_id, day), cfg.health)
    if weather is None and cfg.weather.enabled and cfg.weather.latitude is not None:
        weather = fetch_weather(cfg.weather.latitude, cfg.weather.longitude or 0.0, day)
    status = evaluate_day(day, values, storage.history(hive_id, day), cfg.health, weather)
    storage.save_day(hive_id, status, dict(values))
    return status


def recompute_all(storage: Storage, cfg: Config, hive_id: str | None = None) -> list[DayStatus]:
    hive_id = hive_id or cfg.hive.id
    return [compute_day(storage, cfg, d, hive_id) for d in storage.days_with_data(hive_id)]


def describe(status: DayStatus, lang: str) -> list[str]:
    if status.level == "learning":
        return [t("learning", lang)]
    if not status.reasons:
        return [t("all_fine", lang)]
    return [t(r, lang) for r in status.reasons]


def notify_status(notifier: Notifier, cfg: Config, status: DayStatus) -> bool:
    if status.level not in cfg.alerts.send_status_levels and not status.open_hive:
        return False
    lang = cfg.language
    alert = Alert(
        title=t("alert.status.title", lang, hive=cfg.hive.name, level=t(f"level.{status.level}", lang)),
        body="\n".join(describe(status, lang)),
        priority=4 if status.open_hive else 3,
        tags=["bee"],
        key=f"status:{cfg.hive.id}:{status.day}",
    )
    return notifier.send(alert, cooldown_s=12 * 3600)
