"""Evaluate the daily-status logic on simulated colonies.

* False alarms: healthy colonies simulated for a season (180 days, many seeds);
  count the days that say "open this hive" (2 consecutive check days) and the
  days with any "check".
* Detection delay: for each failure scenario, the number of days between the
  onset of the problem and the first "check" on the right signal, and the first
  "open this hive".
* Ablation: the same with a fixed threshold instead of the hive's own baseline
  (a population-level rule: e.g. pollen share < 10 %), to show why the baseline
  matters when colonies differ in strength.

The simulator is plausible, not measured; these numbers test the logic, not
the biology. Writes docs/results/health.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from smarthive.health import DayRecord, daily_values, evaluate_day, to_record
from smarthive.simulate import simulate

SCENARIO_SIGNAL = {
    "forager_loss": "balance",
    "queen_loss": "pollen_share",
    "robbing": "agitation",
    "swarming": "crowding",
}
# A fixed rule a beekeeper might write without per-hive baselines.
FIXED = {
    "activity": lambda v: v is not None and v < 400,
    "balance": lambda v: v is not None and v > 0.08,
    "pollen_share": lambda v: v is not None and v < 0.10,
    "crowding": lambda v: v is not None and v > 4.0,
    "agitation": lambda v: v is not None and v > 0.10,
}


def run(days, scenario, seed, onset=None, peak=1400.0):
    hist: list[DayRecord] = []
    out = []
    for sd in simulate(days=days, scenario=scenario, seed=seed, onset=onset, peak_traffic=peak):
        values = daily_values(sd.rows)
        st = evaluate_day(sd.day, values, hist, weather=sd.weather)
        hist.append(to_record(st, values))
        out.append((sd, values, st))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--out", type=Path, default=Path("docs/results/health.json"))
    args = ap.parse_args(argv)

    # 1. Healthy colonies of different strengths (peak traffic 300 to 2500 passes/h).
    open_days = check_days = fixed_days = total = 0
    for seed in range(args.seeds):
        peak = 300 + (2200 * seed / max(1, args.seeds - 1))
        for sd, values, st in run(180, "healthy", seed, peak=peak):
            if st.level == "learning":
                continue
            total += 1
            open_days += st.open_hive
            check_days += st.level == "check"
            fixed_days += any(f(values.get(k)) for k, f in FIXED.items()) and not (
                sd.weather["max_temp_c"] < 12 or sd.weather["rain_mm"] >= 2
            )
    healthy = {
        "colony_days": total,
        "open_hive_days_per_season_180d": 180 * open_days / total,
        "check_days_per_season_180d": 180 * check_days / total,
        "fixed_threshold_alarm_days_per_season_180d": 180 * fixed_days / total,
    }
    print("healthy:", json.dumps(healthy))

    # 2. Delay on failure scenarios (onset at day 40 of 60).
    scenarios = {}
    for scenario, signal in SCENARIO_SIGNAL.items():
        delays_check, delays_open, missed = [], [], 0
        fixed_delays, fixed_missed = [], 0
        for seed in range(args.seeds):
            peak = 300 + (2200 * seed / max(1, args.seeds - 1))
            res = run(60, scenario, seed, onset=40, peak=peak)
            first_check = next(
                (i for i, (_, _, st) in enumerate(res) if i >= 40 and st.signals[signal].level == "check"), None
            )
            first_open = next((i for i, (_, _, st) in enumerate(res) if i >= 40 and st.open_hive), None)
            first_fixed = next((i for i, (_, v, _) in enumerate(res) if i >= 40 and FIXED[signal](v.get(signal))), None)
            if first_check is None:
                missed += 1
            else:
                delays_check.append(first_check - 40)
            if first_open is not None:
                delays_open.append(first_open - 40)
            if first_fixed is None:
                fixed_missed += 1
            else:
                fixed_delays.append(first_fixed - 40)
        scenarios[scenario] = {
            "signal": signal,
            "runs": args.seeds,
            "detected": args.seeds - missed,
            "median_days_to_check": statistics.median(delays_check) if delays_check else None,
            "median_days_to_open_hive": statistics.median(delays_open) if delays_open else None,
            "fixed_threshold_detected": args.seeds - fixed_missed,
            "fixed_threshold_median_days": statistics.median(fixed_delays) if fixed_delays else None,
        }
        print(scenario, json.dumps(scenarios[scenario]))

    # 3. Hornet scenario: days until hornet_level == check.
    hornet_delays = []
    for seed in range(args.seeds):
        res = run(60, "hornet", seed, onset=40)
        first = next((i for i, (_, _, st) in enumerate(res) if i >= 40 and st.hornet_level == "check"), None)
        if first is not None:
            hornet_delays.append(first - 40)
    scenarios["hornet"] = {
        "runs": args.seeds,
        "detected": len(hornet_delays),
        "median_days_to_check": statistics.median(hornet_delays) if hornet_delays else None,
    }
    print("hornet", json.dumps(scenarios["hornet"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"healthy": healthy, "scenarios": scenarios}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
