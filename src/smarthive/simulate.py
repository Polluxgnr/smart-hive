"""Synthetic hive history: realistic hourly aggregates with known problems.

Used to try the dashboard and alerts without hardware, and to measure the
status logic (false alarms on healthy colonies, delay on real problems).
The numbers are plausible, not measured: traffic follows a daily bell curve
scaled by temperature and rain, with day-to-day noise.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date, timedelta

from .health import HourRow

SCENARIOS = ("healthy", "forager_loss", "queen_loss", "robbing", "swarming", "hornet")


@dataclass
class SimDay:
    day: str
    rows: list[HourRow]
    weather: dict[str, float]
    problem: bool  # ground truth: is the scenario's problem active this day?
    hornet_events: list[tuple[int, float, float]]  # (hour, duration_s, confidence)


def _ramp(i: int, onset: int, days: int = 4) -> float:
    """0 before onset, rising linearly to 1 over ``days``."""
    if i < onset:
        return 0.0
    return min(1.0, (i - onset + 1) / days)


def simulate(
    days: int = 30,
    scenario: str = "healthy",
    onset: int | None = None,
    seed: int = 0,
    end: date | None = None,
    peak_traffic: float = 1400.0,
) -> list[SimDay]:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
    rng = random.Random(seed)
    end = end or date.today()
    start = end - timedelta(days=days - 1)
    onset = days - 7 if onset is None else onset
    strength = peak_traffic * rng.uniform(0.7, 1.3)
    temp = rng.uniform(19, 25)
    out: list[SimDay] = []
    for i in range(days):
        d = start + timedelta(days=i)
        temp = 0.7 * temp + 0.3 * rng.gauss(22, 4)
        rain = rng.expovariate(1 / 6.0) if rng.random() < 0.18 else 0.0
        weather = {"max_temp_c": round(temp, 1), "rain_mm": round(rain, 1)}
        weather_factor = max(0.05, min(1.2, (temp - 9) / 13)) * (0.35 if rain >= 2 else 1.0)
        day_noise = math.exp(rng.gauss(0, 0.10))
        r = _ramp(i, onset) if scenario != "healthy" else 0.0

        activity_mult, balance, pollen, crowd, erratic, hornet_rate = 1.0, 0.0, 0.18, 1.5, 0.04, 0.0
        if scenario == "forager_loss":
            balance += 0.15 * r
            activity_mult *= 1 - 0.35 * r
        elif scenario == "queen_loss":
            pollen *= 1 - 0.75 * r
            activity_mult *= 1 - 0.15 * r
        elif scenario == "robbing":
            erratic += 0.14 * r
        elif scenario == "swarming":
            crowd += 7.0 * r
        elif scenario == "hornet":
            hornet_rate = 5.0 * r
            activity_mult *= 1 - 0.25 * r
        pollen *= math.exp(rng.gauss(0, 0.12))
        if temp > 29:
            crowd *= 1 + (temp - 29) * 0.4

        rows: list[HourRow] = []
        events: list[tuple[int, float, float]] = []
        for h in range(24):
            shape = math.exp(-(((h - 13.5) / 3.3) ** 2))
            traffic = strength * shape * weather_factor * day_noise * activity_mult
            traffic = max(0, int(rng.gauss(traffic, math.sqrt(traffic + 1))))
            b = balance + rng.gauss(0, 0.012)
            n_out = round(traffic * (1 + b) / 2)
            n_in = max(0, traffic - n_out)
            morning = 1.3 if h < 12 else 0.8
            p_in = min(n_in, round(n_in * pollen * morning * rng.uniform(0.85, 1.15)))
            tracks = int(traffic * 1.4 + rng.uniform(0, 20))
            err = round(tracks * max(0.0, erratic + rng.gauss(0, 0.008)))
            stationary = max(0.0, crowd * shape * rng.uniform(0.7, 1.3)) * 3600 if 7 <= h < 21 else 0.0
            n_hornet = 0
            if hornet_rate > 0 and 10 <= h < 19:
                lam = hornet_rate * math.exp(-(((h - 15) / 2.5) ** 2)) * weather_factor
                n_hornet = _poisson(rng, lam)
                for _ in range(n_hornet):
                    events.append((h, rng.uniform(3, 40), rng.uniform(0.6, 0.97)))
            rows.append(
                HourRow(
                    hour=h,
                    observed_s=3600.0,
                    bees_in=n_in,
                    bees_out=n_out,
                    pollen_in=p_in,
                    tracks=tracks,
                    erratic_tracks=err,
                    stationary_bee_s=stationary,
                    velutina_events=n_hornet,
                )
            )
        out.append(SimDay(d.isoformat(), rows, weather, scenario != "healthy" and r > 0, events))
    return out


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    # Knuth's method is fine for the small rates used here.
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1
