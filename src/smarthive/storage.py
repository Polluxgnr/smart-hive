"""SQLite storage: hourly aggregates, hornet events and daily statuses.

One file (``<data_dir>/smarthive.db``), no server. Hourly rows are *added to*
(upsert with sums), so several runs in the same hour accumulate correctly.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from .health import DayRecord, DayStatus, HourRow

SCHEMA = """
CREATE TABLE IF NOT EXISTS hourly (
    hive_id TEXT NOT NULL,
    day TEXT NOT NULL,          -- YYYY-MM-DD local
    hour INTEGER NOT NULL,      -- 0-23 local
    observed_s REAL DEFAULT 0,
    bees_in INTEGER DEFAULT 0,
    bees_out INTEGER DEFAULT 0,
    pollen_in INTEGER DEFAULT 0,
    tracks INTEGER DEFAULT 0,
    erratic_tracks INTEGER DEFAULT 0,
    stationary_bee_s REAL DEFAULT 0,
    velutina_events INTEGER DEFAULT 0,
    other_vespid_events INTEGER DEFAULT 0,
    simulated INTEGER DEFAULT 0,
    PRIMARY KEY (hive_id, day, hour)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hive_id TEXT NOT NULL,
    ts TEXT NOT NULL,           -- ISO local time
    kind TEXT NOT NULL,         -- 'hawking'
    species TEXT,
    confidence REAL,
    duration_s REAL,
    strikes INTEGER DEFAULT 0,
    snapshot TEXT,              -- path relative to data_dir
    alerted INTEGER DEFAULT 0,
    simulated INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS daily (
    hive_id TEXT NOT NULL,
    day TEXT NOT NULL,
    level TEXT NOT NULL,
    open_hive INTEGER NOT NULL,
    hornet_level TEXT,
    anomalous INTEGER NOT NULL,
    values_json TEXT NOT NULL,
    signals_json TEXT NOT NULL,
    reasons_json TEXT NOT NULL,
    PRIMARY KEY (hive_id, day)
);
"""

HOUR_FIELDS = (
    "observed_s",
    "bees_in",
    "bees_out",
    "pollen_in",
    "tracks",
    "erratic_tracks",
    "stationary_bee_s",
    "velutina_events",
    "other_vespid_events",
)


class Storage:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "snapshots").mkdir(exist_ok=True)
        self.db = sqlite3.connect(self.data_dir / "smarthive.db")
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # -- hourly --------------------------------------------------------------
    def add_hour(self, hive_id: str, day: str, row: HourRow, simulated: bool = False) -> None:
        vals = [getattr(row, f) for f in HOUR_FIELDS]
        updates = ", ".join(f"{f} = {f} + excluded.{f}" for f in HOUR_FIELDS)
        self.db.execute(
            f"INSERT INTO hourly (hive_id, day, hour, {', '.join(HOUR_FIELDS)}, simulated) "
            f"VALUES (?, ?, ?, {', '.join('?' * len(HOUR_FIELDS))}, ?) "
            f"ON CONFLICT(hive_id, day, hour) DO UPDATE SET {updates}",
            [hive_id, day, row.hour, *vals, int(simulated)],
        )
        self.db.commit()

    def hours(self, hive_id: str, day: str) -> list[HourRow]:
        cur = self.db.execute(
            f"SELECT hour, {', '.join(HOUR_FIELDS)} FROM hourly WHERE hive_id = ? AND day = ? ORDER BY hour",
            (hive_id, day),
        )
        return [HourRow(**dict(r)) for r in cur]

    def days_with_data(self, hive_id: str) -> list[str]:
        cur = self.db.execute("SELECT DISTINCT day FROM hourly WHERE hive_id = ? ORDER BY day", (hive_id,))
        return [r[0] for r in cur]

    def hives(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT DISTINCT hive_id FROM hourly ORDER BY hive_id")]

    def is_simulated(self, hive_id: str) -> bool:
        r = self.db.execute("SELECT MAX(simulated) FROM hourly WHERE hive_id = ?", (hive_id,)).fetchone()
        return bool(r and r[0])

    # -- events --------------------------------------------------------------
    def add_event(
        self,
        hive_id: str,
        ts: str,
        species: str,
        confidence: float,
        duration_s: float,
        strikes: int = 0,
        snapshot: str | None = None,
        alerted: bool = False,
        simulated: bool = False,
        kind: str = "hawking",
    ) -> int:
        cur = self.db.execute(
            "INSERT INTO events (hive_id, ts, kind, species, confidence, duration_s, strikes, snapshot, "
            "alerted, simulated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (hive_id, ts, kind, species, confidence, duration_s, strikes, snapshot, int(alerted), int(simulated)),
        )
        self.db.commit()
        return int(cur.lastrowid or 0)

    def events(self, hive_id: str, since_day: str | None = None, limit: int = 50) -> list[dict]:
        q = "SELECT * FROM events WHERE hive_id = ?"
        args: list = [hive_id]
        if since_day:
            q += " AND ts >= ?"
            args.append(since_day)
        q += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.db.execute(q, args)]

    # -- daily ---------------------------------------------------------------
    def save_day(self, hive_id: str, status: DayStatus, values: dict) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO daily VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                hive_id,
                status.day,
                status.level,
                int(status.open_hive),
                status.hornet_level,
                int(status.anomalous),
                json.dumps(values),
                json.dumps({k: asdict(v) for k, v in status.signals.items()}),
                json.dumps(status.reasons),
            ),
        )
        self.db.commit()

    def history(self, hive_id: str, before_day: str) -> list[DayRecord]:
        cur = self.db.execute(
            "SELECT day, values_json, signals_json, anomalous FROM daily WHERE hive_id = ? AND day < ? ORDER BY day",
            (hive_id, before_day),
        )
        out = []
        for r in cur:
            signals = json.loads(r["signals_json"])
            out.append(
                DayRecord(
                    r["day"],
                    json.loads(r["values_json"]),
                    {k: v["level"] for k, v in signals.items()},
                    bool(r["anomalous"]),
                )
            )
        return out

    def daily_rows(self, hive_id: str, limit: int = 30) -> list[dict]:
        cur = self.db.execute("SELECT * FROM daily WHERE hive_id = ? ORDER BY day DESC LIMIT ?", (hive_id, limit))
        rows = [dict(r) for r in cur]
        for r in rows:
            r["values"] = json.loads(r.pop("values_json"))
            r["signals"] = json.loads(r.pop("signals_json"))
            r["reasons"] = json.loads(r.pop("reasons_json"))
        return list(reversed(rows))

    def clear_hive(self, hive_id: str) -> None:
        for table in ("hourly", "events", "daily"):
            self.db.execute(f"DELETE FROM {table} WHERE hive_id = ?", (hive_id,))
        self.db.commit()


def hour_rows_from_dicts(rows: Iterable[dict]) -> list[HourRow]:
    return [HourRow(**{k: v for k, v in r.items() if k in HourRow.__dataclass_fields__}) for r in rows]
