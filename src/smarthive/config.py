"""Configuration: one YAML file per installation (see config.example.yaml).

Coordinates (entrance zone, line, hawking zone) are *normalised* to 0-1 so the
same config works whatever the camera resolution. Secrets are never stored in
the file: the SMTP password comes from the SMART_HIVE_SMTP_PASSWORD variable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .health import HealthConfig

NormPoints = list[list[float]]


@dataclass
class HiveCfg:
    id: str = "hive1"
    name: str = "Ruche 1"


@dataclass
class CameraCfg:
    source: Any = 0  # camera index, video file or rtsp:// URL
    width: int = 1280
    height: int = 720
    process_fps: float = 10.0  # frames analysed per second (the rest are skipped)


@dataclass
class EntranceCfg:
    mode: str = "zone"  # "zone" or "line"
    zone: NormPoints = field(default_factory=lambda: [[0.10, 0.30], [0.90, 0.30], [0.90, 0.42], [0.10, 0.42]])
    line: NormPoints = field(default_factory=lambda: [[0.05, 0.45], [0.95, 0.45]])
    inside_point: list[float] = field(default_factory=lambda: [0.5, 0.2])
    margin: float = 0.015  # line hysteresis, fraction of frame height
    hawk_zone: NormPoints | None = None  # None = whole frame


@dataclass
class ModelsCfg:
    detector: str = "models/detector.pt"
    classifier: str = "models/classifier.pt"
    imgsz: int = 640
    conf: float = 0.25
    classify_every: int = 3  # classify each track every N analysed frames
    tracker: str = "bytetrack"


@dataclass
class HornetCfg:
    hawk_min_seconds: float = 3.0
    max_spread: float = 0.30
    alert_species: list[str] = field(default_factory=lambda: ["vespa_velutina"])
    cooldown_minutes: float = 15.0
    check_per_hour: float = 3.0


@dataclass
class NtfyCfg:
    enabled: bool = False
    server: str = "https://ntfy.sh"
    topic: str = ""


@dataclass
class EmailCfg:
    enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    starttls: bool = True
    username: str = ""
    sender: str = ""
    to: list[str] = field(default_factory=list)


@dataclass
class AlertsCfg:
    ntfy: NtfyCfg = field(default_factory=NtfyCfg)
    email: EmailCfg = field(default_factory=EmailCfg)
    daily_summary_hour: int = 20
    send_status_levels: list[str] = field(default_factory=lambda: ["check"])


@dataclass
class WeatherCfg:
    enabled: bool = False
    latitude: float | None = None
    longitude: float | None = None


@dataclass
class StorageCfg:
    data_dir: str = "hive_data"
    save_annotated_video: bool = False


@dataclass
class DashboardCfg:
    host: str = "0.0.0.0"
    port: int = 8080
    refresh_minutes: float = 5.0


@dataclass
class Config:
    hive: HiveCfg = field(default_factory=HiveCfg)
    language: str = "fr"
    camera: CameraCfg = field(default_factory=CameraCfg)
    entrance: EntranceCfg = field(default_factory=EntranceCfg)
    models: ModelsCfg = field(default_factory=ModelsCfg)
    health: HealthConfig = field(default_factory=HealthConfig)
    hornet: HornetCfg = field(default_factory=HornetCfg)
    alerts: AlertsCfg = field(default_factory=AlertsCfg)
    weather: WeatherCfg = field(default_factory=WeatherCfg)
    storage: StorageCfg = field(default_factory=StorageCfg)
    dashboard: DashboardCfg = field(default_factory=DashboardCfg)

    def to_dict(self) -> dict:
        return asdict(self)


def _build(cls: type, data: dict | None) -> Any:
    data = data or {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown config keys in {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    default = cls()
    for name, value in data.items():
        current = getattr(default, name)
        if is_dataclass(current) and isinstance(value, dict):
            kwargs[name] = _build(type(current), value)
        elif isinstance(current, tuple) and isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None) -> Config:
    if path is None:
        return Config()
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = _build(Config, data)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if cfg.entrance.mode not in ("zone", "line"):
        raise ValueError("entrance.mode must be 'zone' or 'line'")
    pts = [*cfg.entrance.zone, *cfg.entrance.line, cfg.entrance.inside_point]
    if cfg.entrance.hawk_zone:
        pts += cfg.entrance.hawk_zone
    for p in pts:
        if len(p) != 2 or not all(-0.01 <= float(v) <= 1.01 for v in p):
            raise ValueError(f"coordinates must be normalised 0-1 pairs, got {p}")
    if cfg.language not in ("fr", "en"):
        raise ValueError("language must be 'fr' or 'en'")
    if cfg.alerts.ntfy.enabled and not cfg.alerts.ntfy.topic:
        raise ValueError("alerts.ntfy.topic is required when ntfy is enabled")
    if cfg.alerts.email.enabled and not (cfg.alerts.email.smtp_host and cfg.alerts.email.to):
        raise ValueError("alerts.email needs smtp_host and at least one 'to' address")


def save_config(cfg: Config, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False, allow_unicode=True)
