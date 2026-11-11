"""Static HTML dashboard (inline SVG, no external JavaScript) and a tiny web server.

The page is regenerated from SQLite and works offline on a Raspberry Pi.
Charts carry native SVG tooltips (<title>) and every chart has a table view.
"""

from __future__ import annotations

import html
import http.server
import shutil
import socketserver
import statistics
from datetime import date, datetime, timedelta
from functools import partial
from pathlib import Path

from .config import Config
from .health import SIGNALS
from .i18n import t
from .storage import Storage

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--s1:#2a78d6;--s2:#eb6834;--hornet:#e34948;--band:#e1e0d9;
--good:#0ca30c;--warning:#fab219;--critical:#d03b3b;--learning:#898781;--card-border:#e1e0d9}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--page:#0d0d0d;
--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#d95926;
--hornet:#e66767;--band:#2c2c2a;--card-border:#2c2c2a}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#d95926;--hornet:#e66767;--band:#2c2c2a;--card-border:#2c2c2a}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:16px/1.45 system-ui,-apple-system,"Segoe UI",
Roboto,sans-serif;font-variant-numeric:tabular-nums}
main{max-width:1100px;margin:0 auto;padding:20px 16px 48px}
h1{font-size:1.6rem;margin:0 0 4px}h2{font-size:1.15rem;margin:28px 0 10px}h3{font-size:.95rem;margin:0 0 6px}
.sub{color:var(--ink2);font-size:.9rem}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;border:1px solid var(--card-border);
font-size:.8rem;color:var(--ink2);margin-left:8px}
.card{background:var(--surface);border:1px solid var(--card-border);border-radius:12px;padding:16px;margin:12px 0}
.status{display:flex;gap:16px;align-items:flex-start;border-left:8px solid var(--lvl)}
.status .icon{font-size:2rem;line-height:1}
.status .lvl{font-size:1.4rem;font-weight:700}
.status ul{margin:6px 0 0;padding-left:18px}
.open{background:var(--critical);color:#fff;border-radius:8px;padding:8px 12px;font-weight:700;margin-top:8px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--card-border);border-radius:12px;padding:12px}
.tile .v{font-size:1.8rem;font-weight:700}.tile .k{color:var(--ink2);font-size:.85rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
svg{width:100%;height:auto;display:block}
svg text{fill:var(--muted);font-size:11px}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--ink2);font-size:.85rem;margin:4px 0 8px}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-1px;margin-right:6px}
details{margin-top:8px;color:var(--ink2);font-size:.85rem}
table{border-collapse:collapse;width:100%;font-size:.85rem}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}
.events{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
.ev img{width:100%;border-radius:8px;display:block;aspect-ratio:16/9;object-fit:cover}
.ev .meta{font-size:.85rem;color:var(--ink2);margin-top:6px}
footer{margin-top:40px;color:var(--muted);font-size:.8rem}
@media (max-width:600px){h1{font-size:1.3rem}.tile .v{font-size:1.4rem}}
"""

LEVEL_STYLE = {
    "ok": ("var(--good)", "✓"),
    "watch": ("var(--warning)", "!"),
    "check": ("var(--critical)", "⚠"),
    "learning": ("var(--learning)", "…"),
}


def _e(s: object) -> str:
    return html.escape(str(s))


def _fmt(v: float | None, name: str) -> str:
    if v is None:
        return "-"
    if name in ("balance",):
        return f"{v:+.1%}"
    if name in ("pollen_share", "agitation"):
        return f"{v:.0%}"
    if name == "crowding":
        return f"{v:.1f}"
    return f"{v:,.0f}".replace(",", " ")


def _traffic_svg(
    today: list[float], base_med: list[float], base_lo: list[float], base_hi: list[float], lang: str
) -> str:
    w, h, left, bottom, top = 640, 220, 40, 24, 10
    ymax = max([*today, *base_hi, 10]) * 1.1
    bw = (w - left - 8) / 24

    def y(v: float) -> float:
        return top + (h - top - bottom) * (1 - v / ymax)

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{_e(t("dash.traffic", lang))}">']
    for k in range(5):
        v = ymax * k / 4
        parts.append(
            f'<line x1="{left}" x2="{w - 8}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="var(--grid)" '
            f'stroke-width="1"/><text x="{left - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.0f}</text>'
        )
    # Baseline band (25-75 %) and median line.
    if any(base_hi):
        pts_hi = [(left + bw * (i + 0.5), y(v)) for i, v in enumerate(base_hi)]
        pts_lo = [(left + bw * (i + 0.5), y(v)) for i, v in enumerate(base_lo)]
        poly = " ".join(f"{x:.1f},{yy:.1f}" for x, yy in pts_hi + pts_lo[::-1])
        parts.append(f'<polygon points="{poly}" fill="var(--band)" opacity="0.8"/>')
        med = " ".join(f"{left + bw * (i + 0.5):.1f},{y(v):.1f}" for i, v in enumerate(base_med))
        parts.append(
            f'<polyline points="{med}" fill="none" stroke="var(--muted)" stroke-width="2" stroke-dasharray="4 3"/>'
        )
    for i, v in enumerate(today):
        x = left + bw * i + 2
        bh = max(0.0, y(0) - y(v))
        parts.append(
            f"<g><title>{i:02d}:00 - {v:.0f} ({t('dash.baseline', lang)} {base_med[i]:.0f})</title>"
            f'<rect x="{x:.1f}" y="{y(v):.1f}" width="{bw - 4:.1f}" height="{bh:.1f}" rx="2" fill="var(--s1)"/>'
            f'<rect x="{left + bw * i:.1f}" y="{top}" width="{bw:.1f}" height="{h - top - bottom}" fill="transparent"/>'
            f"</g>"
        )
        if i % 3 == 0:
            parts.append(f'<text x="{left + bw * (i + 0.5):.1f}" y="{h - 6}" text-anchor="middle">{i}h</text>')
    parts.append(f'<line x1="{left}" x2="{w - 8}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="var(--axis)"/>')
    parts.append("</svg>")
    return "".join(parts)


def _history_svg(
    days: list[str], values: list[float | None], medians: list[float | None], levels: list[str], name: str
) -> str:
    w, h, left, bottom, top = 320, 120, 36, 18, 8
    vals = [v for v in values + medians if v is not None]
    if not vals:
        return '<p class="sub">-</p>'
    lo, hi = min(vals), max(vals)
    if hi == lo:
        hi = lo + 1
    pad = (hi - lo) * 0.1
    lo, hi = lo - pad, hi + pad
    n = len(days)

    def x(i: int) -> float:
        return left + (w - left - 6) * (i / max(1, n - 1))

    def y(v: float) -> float:
        return top + (h - top - bottom) * (1 - (v - lo) / (hi - lo))

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img">']
    for v in (lo + pad, hi - pad):
        parts.append(
            f'<line x1="{left}" x2="{w - 6}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="var(--grid)"/>'
            f'<text x="{left - 4}" y="{y(v) + 4:.1f}" text-anchor="end">{_fmt(v, name)}</text>'
        )
    med_pts = [(x(i), y(m)) for i, m in enumerate(medians) if m is not None]
    if len(med_pts) > 1:
        parts.append(
            '<polyline points="'
            + " ".join(f"{a:.1f},{b:.1f}" for a, b in med_pts)
            + '" fill="none" stroke="var(--muted)" stroke-width="1.5" stroke-dasharray="4 3"/>'
        )
    seg: list[str] = []
    for i, v in enumerate(values):
        if v is None:
            if len(seg) > 1:
                parts.append(f'<polyline points="{" ".join(seg)}" fill="none" stroke="var(--s1)" stroke-width="2"/>')
            seg = []
            continue
        seg.append(f"{x(i):.1f},{y(v):.1f}")
    if len(seg) > 1:
        parts.append(f'<polyline points="{" ".join(seg)}" fill="none" stroke="var(--s1)" stroke-width="2"/>')
    for i, (d, v, lv) in enumerate(zip(days, values, levels, strict=True)):
        if v is None:
            continue
        colour = {"watch": "var(--warning)", "check": "var(--critical)"}.get(lv)
        r = 4.5 if colour else 2.5
        fill = colour or "var(--s1)"
        tip = f"{d}: {_fmt(v, name)}" + (f" ({lv})" if colour else "")
        if colour:
            # Shape as well as colour: triangle = watch, diamond = check.
            cx, cy = x(i), y(v)
            shape = (
                f"{cx:.1f},{cy - 6:.1f} {cx + 5.5:.1f},{cy + 4:.1f} {cx - 5.5:.1f},{cy + 4:.1f}"
                if lv == "watch"
                else f"{cx:.1f},{cy - 6:.1f} {cx + 6:.1f},{cy:.1f} {cx:.1f},{cy + 6:.1f} {cx - 6:.1f},{cy:.1f}"
            )
            parts.append(
                f'<g><title>{_e(tip)}</title><polygon points="{shape}" fill="{fill}" '
                f'stroke="var(--surface)" stroke-width="2"/></g>'
            )
        else:
            parts.append(
                f'<g><title>{_e(tip)}</title><circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="8" '
                f'fill="transparent"/><circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="{r}" fill="{fill}"/></g>'
            )
    parts.append(
        f'<text x="{left}" y="{h - 4}">{_e(days[0][5:])}</text>'
        f'<text x="{w - 6}" y="{h - 4}" text-anchor="end">{_e(days[-1][5:])}</text></svg>'
    )
    return "".join(parts)


def _hornet_svg(days: list[str], counts: list[int], lang: str) -> str:
    w, h, left, bottom, top = 640, 120, 30, 18, 8
    ymax = max([*counts, 3])
    bw = (w - left - 6) / max(1, len(days))
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{_e(t("dash.hornets", lang))}">']
    parts.append(f'<text x="{left - 4}" y="{top + 8}" text-anchor="end">{ymax}</text>')
    for i, (d, c) in enumerate(zip(days, counts, strict=True)):
        bh = (h - top - bottom) * c / ymax
        parts.append(
            f'<g><title>{d}: {c}</title><rect x="{left + bw * i:.1f}" y="{top}" width="{bw:.1f}" '
            f'height="{h - top - bottom}" fill="transparent"/>'
            f'<rect x="{left + bw * i + 1:.1f}" y="{h - bottom - bh:.1f}" width="{max(1.0, bw - 2):.1f}" '
            f'height="{bh:.1f}" rx="2" fill="var(--hornet)"/></g>'
        )
    parts.append(f'<line x1="{left}" x2="{w - 6}" y1="{h - bottom}" y2="{h - bottom}" stroke="var(--axis)"/>')
    parts.append(
        f'<text x="{left}" y="{h - 4}">{_e(days[0][5:])}</text>'
        f'<text x="{w - 6}" y="{h - 4}" text-anchor="end">{_e(days[-1][5:])}</text></svg>'
    )
    return "".join(parts)


def _hourly_baseline(storage: Storage, hive: str, day: str, window: int = 21):
    past = [d for d in storage.days_with_data(hive) if d < day][-window:]
    per_hour: list[list[float]] = [[] for _ in range(24)]
    for d in past:
        for r in storage.hours(hive, d):
            per_hour[r.hour].append((r.bees_in + r.bees_out) * 3600 / max(r.observed_s, 1))
    med, lo, hi = [], [], []
    for vals in per_hour:
        if len(vals) >= 3:
            q = statistics.quantiles(vals, n=4)
            med.append(statistics.median(vals))
            lo.append(q[0])
            hi.append(q[2])
        else:
            med.append(0.0)
            lo.append(0.0)
            hi.append(0.0)
    return med, lo, hi


def render(storage: Storage, cfg: Config, out_dir: str | Path, day: str | None = None) -> Path:
    """Write ``index.html`` (and copy snapshots) into ``out_dir``. Returns the page path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lang = cfg.language
    hives = storage.hives() or [cfg.hive.id]
    body: list[str] = []
    for hive in hives:
        body.append(_render_hive(storage, cfg, hive, out, day))
    page = (
        f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<meta http-equiv="refresh" content="{int(cfg.dashboard.refresh_minutes * 60)}">'
        f"<title>Smart Hive</title><style>{CSS}</style></head><body><main>"
        f"<h1>{_e(t('dash.title', lang))}</h1>"
        f'<div class="sub">{_e(t("dash.updated", lang))} {datetime.now():%Y-%m-%d %H:%M}</div>'
        + "".join(body)
        + "<footer>Smart Hive - AGPL-3.0. Local only: nothing is sent unless alerts are configured.</footer>"
        "</main></body></html>"
    )
    path = out / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def _render_hive(storage: Storage, cfg: Config, hive: str, out: Path, day: str | None) -> str:
    lang = cfg.language
    rows = storage.daily_rows(hive, limit=30)
    days_all = storage.days_with_data(hive)
    day = day or (days_all[-1] if days_all else date.today().isoformat())
    today_row = next((r for r in rows if r["day"] == day), rows[-1] if rows else None)
    name = cfg.hive.name if hive == cfg.hive.id else hive
    sim = storage.is_simulated(hive)
    parts = [f"<h2>{_e(name)}{'<span class=badge>' + _e(t('dash.simulated', lang)) + '</span>' if sim else ''}</h2>"]

    # -- status card ---------------------------------------------------------
    if today_row:
        lvl = today_row["level"]
        colour, icon = LEVEL_STYLE.get(lvl, LEVEL_STYLE["learning"])
        reasons = today_row["reasons"] or (["learning"] if lvl == "learning" else ["all_fine"])
        items = "".join(f"<li>{_e(t(r, lang))}</li>" for r in reasons if r != "open_hive")
        open_banner = f'<div class="open">{_e(t("open_hive", lang))}</div>' if today_row["open_hive"] else ""
        parts.append(
            f'<div class="card status" style="--lvl:{colour}"><div class="icon" style="color:{colour}">{icon}</div>'
            f'<div><div class="sub">{_e(t("dash.today", lang))} - {_e(day)}</div>'
            f'<div class="lvl">{_e(t("level." + lvl, lang))}</div>{open_banner}<ul>{items}</ul></div></div>'
        )

    # -- tiles ---------------------------------------------------------------
    hours = storage.hours(hive, day)
    b_in = sum(r.bees_in for r in hours)
    b_out = sum(r.bees_out for r in hours)
    pollen = sum(r.pollen_in for r in hours)
    events_today = [e for e in storage.events(hive, since_day=day, limit=500) if e["ts"].startswith(day)]
    velutina = sum(1 for e in events_today if e["species"] == "vespa_velutina")
    tiles = [
        (f"{b_in:,}".replace(",", " "), t("dash.in", lang)),
        (f"{b_out:,}".replace(",", " "), t("dash.out", lang)),
        (f"{pollen / b_in:.0%}" if b_in else "-", t("dash.pollen", lang)),
        (str(velutina), t("species.vespa_velutina", lang)),
    ]
    parts.append(
        '<div class="tiles">'
        + "".join(f'<div class="tile"><div class="v">{_e(v)}</div><div class="k">{_e(k)}</div></div>' for v, k in tiles)
        + "</div>"
    )

    # -- traffic today vs baseline -------------------------------------------
    today = [0.0] * 24
    for r in hours:
        today[r.hour] = (r.bees_in + r.bees_out) * (3600 / r.observed_s if r.observed_s >= 600 else 1)
    med, lo, hi = _hourly_baseline(storage, hive, day)
    parts.append(
        f'<div class="card"><h3>{_e(t("dash.traffic", lang))}</h3>'
        f'<div class="legend"><span><span class="sw" style="background:var(--s1)"></span>'
        f'{_e(t("dash.today_line", lang))}</span><span><span class="sw" style="background:var(--band);'
        f'border:2px dashed var(--muted)"></span>{_e(t("dash.baseline", lang))} (25-75 %)</span></div>'
        + _traffic_svg(today, med, lo, hi, lang)
        + "<details><summary>Table</summary><table><tr><th>h</th><th>"
        + _e(t("dash.today_line", lang))
        + "</th><th>"
        + _e(t("dash.baseline", lang))
        + "</th></tr>"
        + "".join(
            f"<tr><td>{h:02d}</td><td>{today[h]:.0f}</td><td>{med[h]:.0f}</td></tr>"
            for h in range(24)
            if today[h] or med[h]
        )
        + "</table></details></div>"
    )

    # -- 30-day history ------------------------------------------------------
    if rows:
        days = [r["day"] for r in rows]
        cards = []
        for sig in SIGNALS:
            vals = [r["values"].get(sig) for r in rows]
            meds = [r["signals"].get(sig, {}).get("median") for r in rows]
            levels = [r["signals"].get(sig, {}).get("level", "learning") for r in rows]
            cards.append(
                f'<div class="card"><h3>{_e(t("signal." + sig, lang))}</h3>'
                + _history_svg(days, vals, meds, levels, sig)
                + "</div>"
            )
        parts.append(
            f"<h2>{_e(t('dash.history', lang))}</h2>"
            '<div class="legend"><span><span class="sw" style="background:var(--s1)"></span>'
            f"{_e(t('dash.today_line', lang))}</span><span>- - {_e(t('dash.baseline', lang))}</span>"
            f'<span style="color:var(--warning)">▲</span> {_e(t("level.watch", lang))}'
            f'<span style="color:var(--critical)">◆</span> {_e(t("level.check", lang))}</div>'
            f'<div class="grid">{"".join(cards)}</div>'
        )

    # -- hornets ---------------------------------------------------------------
    parts.append(f"<h2>{_e(t('dash.hornets', lang))}</h2>")
    span = [(date.fromisoformat(day) - timedelta(days=29 - i)).isoformat() for i in range(30)]
    all_events = storage.events(hive, since_day=span[0], limit=5000)
    counts = [sum(1 for e in all_events if e["ts"].startswith(d) and e["species"] == "vespa_velutina") for d in span]
    if any(counts):
        per_day = "hawking events per day" if lang == "en" else "vols stationnaires par jour"
        parts.append(
            f'<div class="card"><h3>{_e(t("species.vespa_velutina", lang))}: {per_day}</h3>'
            f"{_hornet_svg(span, counts, lang)}</div>"
        )
    shown = [e for e in all_events if e["snapshot"]][:12]
    if not all_events:
        parts.append(f'<p class="sub">{_e(t("dash.no_hornets", lang))}</p>')
    elif shown:
        snap_dir = out / "snapshots"
        snap_dir.mkdir(exist_ok=True)
        cards = []
        for e in shown:
            src = storage.data_dir / e["snapshot"]
            if src.exists():
                shutil.copy2(src, snap_dir / src.name)
            cards.append(
                f'<div class="card ev"><a href="snapshots/{_e(Path(e["snapshot"]).name)}">'
                f'<img loading="lazy" src="snapshots/{_e(Path(e["snapshot"]).name)}" alt="snapshot"></a>'
                f'<div class="meta"><b>{_e(t("species." + (e["species"] or "vespid"), lang))}</b> '
                f"{e['confidence']:.0%} - {_e(e['ts'].replace('T', ' '))}<br>"
                f"{e['duration_s']:.0f} s</div></div>"
            )
        parts.append(f'<div class="events">{"".join(cards)}</div>')
    return "".join(parts)


def serve(directory: str | Path, host: str = "0.0.0.0", port: int = 8080) -> None:
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    with socketserver.ThreadingTCPServer((host, port), handler) as httpd:
        print(f"Dashboard: http://{'localhost' if host in ('0.0.0.0', '') else host}:{port}/  (Ctrl+C to stop)")
        httpd.serve_forever()
