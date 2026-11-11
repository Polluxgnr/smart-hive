import pytest

from smarthive.health import (
    SIGNALS,
    DayRecord,
    HealthConfig,
    HourRow,
    badness,
    daily_values,
    evaluate_day,
    robust_baseline,
    to_record,
)
from smarthive.simulate import simulate


def history(values: dict, n: int = 14):
    return [DayRecord(f"2026-06-{i + 1:02d}", dict(values), dict.fromkeys(values, "ok")) for i in range(n)]


NORMAL = {"activity": 800.0, "balance": 0.0, "pollen_share": 0.18, "crowding": 1.5, "agitation": 0.04}


def test_robust_baseline_ignores_one_outlier():
    med, sigma = robust_baseline([10, 11, 9, 10, 12, 10, 500], SIGNALS["activity"])
    assert med == 10
    assert sigma < 5


def test_baseline_floor_prevents_division_by_zero():
    _, sigma = robust_baseline([0.18] * 10, SIGNALS["pollen_share"])
    assert sigma >= 0.02


def test_badness_sign_follows_bad_direction():
    assert badness(50, 100, 10, SIGNALS["activity"]) == pytest.approx(5)  # down is bad
    assert badness(0.2, 0.0, 0.05, SIGNALS["balance"]) == pytest.approx(4)  # up is bad


def test_learning_until_min_days():
    st = evaluate_day("2026-06-20", NORMAL, history(NORMAL, n=3))
    assert st.level == "learning"
    assert not st.open_hive


def test_normal_day_is_ok():
    st = evaluate_day("2026-06-20", NORMAL, history(NORMAL))
    assert st.level == "ok"
    assert st.reasons == []


def test_pollen_drop_is_check_and_open_after_two_days():
    bad = {**NORMAL, "pollen_share": 0.03}
    hist = history(NORMAL)
    day1 = evaluate_day("2026-06-20", bad, hist)
    assert day1.level == "check" and "pollen_share_check" in day1.reasons
    assert not day1.open_hive
    day2 = evaluate_day("2026-06-21", bad, [*hist, to_record(day1, bad)])
    assert day2.open_hive
    assert day2.reasons[0] == "open_hive"


def test_anomalous_days_do_not_enter_the_baseline():
    bad = {**NORMAL, "balance": 0.2}
    hist = history(NORMAL)
    st = evaluate_day("2026-06-20", bad, hist)
    assert st.anomalous
    # Ten more bad days: the baseline must not drift towards the bad value.
    for i in range(10):
        hist.append(to_record(st, bad))
        st = evaluate_day(f"2026-07-{i + 1:02d}", bad, hist)
    assert st.signals["balance"].median == pytest.approx(0.0)
    assert st.level == "check"


def test_low_activity_on_a_cold_day_is_explained_by_weather():
    quiet = {**NORMAL, "activity": 150.0}
    st = evaluate_day("2026-06-20", quiet, history(NORMAL), weather={"max_temp_c": 9, "rain_mm": 0})
    assert st.signals["activity"].level == "ok"
    assert st.signals["activity"].note == "weather"
    assert "quiet_weather" in st.reasons
    assert st.anomalous  # cold days are kept out of the baseline


def test_hornet_pressure_uses_fixed_levels():
    st = evaluate_day("2026-06-20", {**NORMAL, "velutina_peak_hour": 1}, history(NORMAL))
    assert st.hornet_level == "present" and st.level == "watch"
    st = evaluate_day("2026-06-20", {**NORMAL, "velutina_peak_hour": 4}, history(NORMAL))
    assert st.hornet_level == "check" and st.level == "check"


def test_daily_values_rates_and_ratios():
    rows = [
        HourRow(
            hour=h,
            observed_s=1800,
            bees_in=100,
            bees_out=110,
            pollen_in=20,
            tracks=300,
            erratic_tracks=15,
            stationary_bee_s=3600,
        )
        for h in range(8, 12)
    ]
    v = daily_values(rows, HealthConfig())
    assert v["activity"] == pytest.approx(840 / 2)  # 840 passes in 2 observed hours
    assert v["balance"] == pytest.approx(40 / 840)
    assert v["pollen_share"] == pytest.approx(0.2)
    assert v["agitation"] == pytest.approx(0.05)
    assert v["crowding"] == pytest.approx(4 * 3600 / (2 * 3600))


def test_daily_values_ratios_need_enough_traffic():
    v = daily_values([HourRow(hour=10, bees_in=5, bees_out=5, tracks=10)])
    assert v["balance"] is None and v["pollen_share"] is None and v["agitation"] is None


def test_night_hours_are_ignored_for_activity():
    v = daily_values([HourRow(hour=2, bees_in=500, bees_out=500)])
    assert v["activity"] is None


@pytest.mark.parametrize(
    "scenario,signal",
    [
        ("forager_loss", "balance"),
        ("queen_loss", "pollen_share"),
        ("robbing", "agitation"),
        ("swarming", "crowding"),
    ],
)
def test_simulated_problems_are_detected(scenario, signal):
    hist: list[DayRecord] = []
    flagged = False
    for sd in simulate(days=40, scenario=scenario, seed=3):
        values = daily_values(sd.rows)
        st = evaluate_day(sd.day, values, hist, weather=sd.weather)
        hist.append(to_record(st, values))
        if sd.problem and st.signals[signal].level == "check":
            flagged = True
    assert flagged


def test_healthy_simulation_rarely_raises_check():
    checks = 0
    for seed in range(5):
        hist: list[DayRecord] = []
        for sd in simulate(days=60, scenario="healthy", seed=seed):
            values = daily_values(sd.rows)
            st = evaluate_day(sd.day, values, hist, weather=sd.weather)
            hist.append(to_record(st, values))
            checks += st.open_hive
    assert checks <= 1
