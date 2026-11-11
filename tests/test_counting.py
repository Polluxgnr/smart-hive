import pytest

from smarthive.counting import EntranceCounter, LineRule, ZoneRule
from smarthive.geometry import point_in_polygon, signed_distance_to_line, turning_index

# Entrance slot zone: a band at the top of the board, y in [0, 20].
ZONE = [(0, 0), (100, 0), (100, 20), (0, 20)]


def run(counter, path, tid=1, t0=0.0, dt=0.1):
    events = []
    for i, p in enumerate(path):
        events += counter.update(t0 + i * dt, {tid: p})
    return events


def test_point_in_polygon():
    assert point_in_polygon((50, 10), ZONE)
    assert not point_in_polygon((50, 30), ZONE)
    assert not point_in_polygon((150, 10), ZONE)


def test_signed_distance_left_of_direction_is_positive():
    # Line drawn left to right; in image coordinates "left" is above (smaller y).
    assert signed_distance_to_line((5, 0), (0, 10), (10, 10)) == pytest.approx(10)
    assert signed_distance_to_line((5, 20), (0, 10), (10, 10)) == pytest.approx(-10)


def test_zone_bee_walking_into_slot_counts_in():
    c = EntranceCounter(ZoneRule(ZONE), max_age=0.5)
    path = [(50, 80), (50, 60), (50, 40), (50, 25), (50, 15), (50, 8)]
    assert run(c, path) == []  # nothing until the track ends
    events = c.flush()
    assert [e.direction for e in events] == ["in"]
    assert (c.total_in, c.total_out) == (1, 0)


def test_zone_bee_leaving_slot_counts_out():
    c = EntranceCounter(ZoneRule(ZONE))
    run(c, [(50, 5), (50, 15), (50, 30), (50, 60)])
    assert [e.direction for e in c.flush()] == ["out"]


def test_zone_walker_passing_by_is_not_counted():
    c = EntranceCounter(ZoneRule(ZONE))
    run(c, [(10, 60), (30, 60), (60, 60), (90, 60)])
    run(c, [(10, 60), (30, 10), (60, 10), (90, 60)], tid=2)  # crosses the zone but leaves again
    run(c, [(10, 10), (30, 12), (60, 10)], tid=3)  # guard staying inside
    assert c.flush() == []


def test_zone_short_tracks_are_ignored():
    c = EntranceCounter(ZoneRule(ZONE), min_points=3)
    run(c, [(50, 40), (50, 10)])
    assert c.flush() == []


def test_zone_track_finishes_after_max_age():
    c = EntranceCounter(ZoneRule(ZONE), max_age=0.5)
    run(c, [(50, 60), (50, 40), (50, 10)])
    assert c.update(0.5, {}) == []  # 0.3 s since last seen: still open
    events = c.update(1.0, {})
    assert [e.direction for e in events] == ["in"]
    assert c.open_tracks == 0


def test_line_rule_counts_both_directions_with_hysteresis():
    rule = LineRule((0, 50), (100, 50), inside_point=(50, 0), margin=5)
    c = EntranceCounter(rule)
    events = run(c, [(50, 80), (50, 60), (50, 52), (50, 48), (50, 40), (50, 20)])
    assert [e.direction for e in events] == ["in"]
    events = run(c, [(50, 20), (50, 60)], tid=2)
    assert [e.direction for e in events] == ["out"]


def test_line_rule_jitter_on_the_line_does_not_double_count():
    rule = LineRule((0, 50), (100, 50), inside_point=(50, 0), margin=5)
    c = EntranceCounter(rule)
    path = [(50, 70), (50, 52), (50, 48), (50, 53), (50, 47), (50, 51), (50, 30)]
    events = run(c, path)
    assert [e.direction for e in events] == ["in"]


def test_line_rule_inside_point_sets_orientation():
    # Same geometry, hive below the line this time.
    rule = LineRule((0, 50), (100, 50), inside_point=(50, 100), margin=5)
    c = EntranceCounter(rule)
    assert [e.direction for e in run(c, [(50, 20), (50, 80)])] == ["in"]


def test_line_rule_rejects_inside_point_on_line():
    with pytest.raises(ValueError):
        LineRule((0, 50), (100, 50), inside_point=(10, 50), margin=5)


def test_multiple_tracks_counted_independently():
    c = EntranceCounter(ZoneRule(ZONE))
    for i in range(5):
        c.update(i * 0.1, {1: (20, 60 - 12 * i), 2: (80, 5 + 12 * i)})
    events = c.flush()
    assert sorted(e.direction for e in events) == ["in", "out"]


def test_turning_index():
    straight = [(i, 0) for i in range(10)]
    zigzag = [(i, (i % 2) * 3) for i in range(10)]
    assert turning_index(straight) == pytest.approx(0)
    assert turning_index(zigzag) > 1.5
