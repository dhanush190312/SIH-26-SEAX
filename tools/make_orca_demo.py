"""Generate ORCA's bundled DEMO datasets (orca/data/demo/*.json).

EVERYTHING GENERATED HERE IS SYNTHETIC. Values are shaped like plausible Bay of
Bengal coastal conditions off Nagapattinam so the pipeline has something honest to
chew on, but none of them is an INCOIS/IMD/ISRO measurement, and the geofence
polygons are illustrative shapes, not official boundaries. Every observation ORCA
builds from these files is stamped ``data_mode=DEMO`` and every source label says
"synthetic demo feed".

Time is stored *relative*: hour 0 = local (IST) midnight today, so the demo always
covers "today" and "tomorrow" whatever day it is run on.

Deterministic: no randomness. Re-running produces byte-identical files.

    python tools/make_orca_demo.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "orca" / "data" / "demo"
HOURS = 48

#: (south, west, north, east) of the Nagapattinam demo area. Outside it, the bundled
#: geofence and advisory layers make NO claim (the pipeline reports INSUFFICIENT DATA).
COVERAGE = [9.6, 79.3, 11.9, 80.95]

HARBOUR = (10.7650, 79.9000)  # WP1 sits just off the harbour mouth


def hav(a, b):
    R = 6371.0088
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    x = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(x))


def interp(keys: dict[int, float], hod: float) -> float:
    ks = sorted(keys)
    if hod <= ks[0]:
        return keys[ks[0]]
    for a, b in zip(ks, ks[1:]):
        if a <= hod <= b:
            return keys[a] + (keys[b] - keys[a]) * (hod - a) / (b - a)
    return keys[ks[-1]]


WAVE = {0: 0.9, 3: 0.85, 6: 0.9, 8: 1.1, 10: 1.4, 12: 1.7, 14: 2.0, 16: 2.1, 18: 1.9, 21: 1.5, 23: 1.2, 24: 0.9}
WIND = {0: 10, 3: 9, 6: 14, 8: 20, 10: 27, 12: 32, 14: 34, 16: 33, 18: 28, 21: 20, 23: 14, 24: 10}

WEATHER_POINTS = [
    ("WP1", 10.7650, 79.9000), ("WP2", 10.9500, 80.0500), ("WP3", 10.7000, 80.1500), ("WP4", 10.4500, 80.0500),
    ("WP5", 10.7200, 80.3200), ("WP6", 11.0000, 80.1200), ("WP7", 10.3800, 80.1800), ("WP8", 10.6000, 80.4000),
    ("WP9", 10.8300, 79.9800), ("WP10", 11.0600, 79.9500), ("WP11", 10.8600, 80.0800),
]
WEATHER_POINTS += [(f"WG{i * 3 + j + 1}", la, lo) for i, la in enumerate((10.87, 10.93, 10.99)) for j, lo in enumerate((79.97, 80.03, 80.09))]
SQUALL = (10.93, 80.03)  # a localised squall band that matures mid-morning


def band(lat, lon, hod, day):
    """0..1 weight of the squall at this place/hour (tomorrow only, a synthetic feature)."""
    if day != 1:
        return 0.0
    d = math.hypot(lat - SQUALL[0], lon - SQUALL[1])
    w_t = max(0.0, 1.0 - abs(hod - 10.5) / 3.0)
    return math.exp(-((d / 0.09) ** 2)) * w_t


def series_for(lat, lon, source):
    wave, wind, rain, lightning = [], [], [], []
    for h in range(HOURS):
        day, hod = divmod(h, 24)
        f = 1.0 + 0.004 * hav(HARBOUR, (lat, lon))
        day_scale = 1.0 if day == 1 else 0.92
        b = band(lat, lon, hod, day)
        w = interp(WAVE, hod) * f * day_scale + 2.2 * b
        v = interp(WIND, hod) * (0.95 + 0.05 * f) * day_scale + 38.0 * b
        r = 0.0
        if day == 1 and 10 <= hod <= 11:
            r = 0.6
        if day == 1 and 15 <= hod <= 17:
            r = 3.0 + 4.5 * b
        lt = 1 if (day == 1 and 16 <= hod <= 17 and lon > 80.10 and lat < 10.80) else 0
        if source == "open_model":
            w = w * 1.06 + 0.03
            v = v * 1.04 + 0.5
            if (lat, lon) == (10.7650, 79.9000) and h == 34:  # 10:00 tomorrow: the disagreement scenario
                w = 2.1
        wave.append(round(w, 2))
        wind.append(round(v, 1))
        rain.append(round(r, 1))
        lightning.append(lt)
    return {"wave_height": wave, "wind_speed": wind, "precipitation": rain, "lightning": lightning}


def weather_doc():
    return {
        "meta": {"kind": "weather", "anchor": "local_midnight_today", "hours": HOURS,
                 "origin": "SYNTHETIC DEMO DATA - not a real forecast"},
        "sources": {
            "incois": {"id": "demo-incois", "label": "INCOIS (synthetic demo feed)", "issued_hours_before_now": 2.1},
            "open_model": {"id": "demo-open-model", "label": "Open model forecast (synthetic demo feed)", "issued_hours_before_now": 3.4},
        },
        "points": [
            {"id": pid, "lat": lat, "lon": lon,
             "series": {src: series_for(lat, lon, src) for src in ("incois", "open_model")}}
            for pid, lat, lon in WEATHER_POINTS
        ],
    }


def ocean_doc():
    # (id, lat, lon, sst, chl, sst gradient per 10 km)
    pts = [
        ("OP-H", 10.7650, 79.9000, 29.1, 0.38, 0.10),
        ("OP-A", 11.0000, 80.1200, 28.6, 0.72, 0.41),
        ("OP-B", 10.7200, 80.3200, 28.9, 0.55, 0.30),
        ("OP-C", 10.3800, 80.1800, 29.3, 0.44, 0.31),
    ]
    def cur(base, ph):
        sp = [round(base + 0.18 * math.sin(2 * math.pi * (h - ph) / 12.42), 2) for h in range(HOURS)]
        return sp
    return {
        "meta": {"kind": "ocean", "anchor": "local_midnight_today", "hours": HOURS,
                 "origin": "SYNTHETIC DEMO DATA - not a satellite retrieval"},
        "sources": {"incois": {"id": "demo-incois-ocean", "label": "INCOIS/ISRO ocean products (synthetic demo feed)",
                               "observed_hours_before_now": 20.5}},
        "points": [
            {"id": i, "lat": la, "lon": lo, "sst": s, "chlorophyll": c, "sst_gradient": g,
             "current_speed": cur(0.35 + 0.05 * k, 2.0 + k)}
            for k, (i, la, lo, s, c, g) in enumerate(pts)
        ],
        "tide": {"station": "Nagapattinam tide station (synthetic)", "lat": 10.7672, "lon": 79.8449,
                 "issued_hours_before_now": 30.0,
                 "series": [round(0.55 + 0.45 * math.sin(2 * math.pi * (h - 3.0) / 12.42), 2) for h in range(HOURS)]},
    }


def pfz_doc():
    return {
        "meta": {"kind": "pfz", "origin": "SYNTHETIC DEMO DATA - not an INCOIS PFZ advisory"},
        "source": {"family": "incois", "id": "demo-incois-pfz", "label": "INCOIS PFZ (synthetic demo advisory)",
                   "issued_hours_before_now": 9.0},
        "valid_day_offsets": [0, 1],
        "zones": [
            {"id": "PFZ-A", "name": "PFZ-A (north-east)", "lat": 11.0000, "lon": 80.1200, "radius_km": 8.0},
            {"id": "PFZ-B", "name": "PFZ-B (east)", "lat": 10.7200, "lon": 80.3200, "radius_km": 9.0},
            {"id": "PFZ-C", "name": "PFZ-C (south-east)", "lat": 10.3800, "lon": 80.1800, "radius_km": 8.0},
            {"id": "PFZ-D", "name": "PFZ-D (inside the illustrative naval area)", "lat": 10.7400, "lon": 80.1400, "radius_km": 6.0},
        ],
    }


def advisory_doc():
    return {
        "meta": {"kind": "advisory", "origin": "SYNTHETIC DEMO DATA - not an IMD bulletin"},
        "source": {"family": "imd", "id": "demo-imd", "label": "IMD marine bulletin (synthetic demo)",
                   "issued_hours_before_now": 4.0},
        "feed_ok": True,
        "coverage": COVERAGE,
        "cyclone": {"status": "none", "text": "No cyclonic system reported near the Tamil Nadu coast (synthetic demo statement)."},
        "items": [
            {"id": "ADV-1", "kind": "small_craft_caution", "severity": "caution",
             "text": "Small-craft caution: strong winds and rough seas off the Nagapattinam-Karaikal coast (synthetic demo advisory).",
             "polygon": [[10.60, 79.90], [10.60, 80.50], [11.20, 80.50], [11.20, 79.90]],
             "from_h": 36, "to_h": 48},
        ],
    }


def geofence_doc():
    return {
        "meta": {"kind": "geofence", "coverage": COVERAGE,
                 "origin": "ILLUSTRATIVE shapes for the prototype - NOT official boundaries. "
                           "Supply official GIS via orca/data/geofences_official.json (see docs/DATA_SOURCES.md)."},
        "geofences": [
            {"id": "GF-NAVAL", "name": "Naval firing-exercise area (illustrative)", "kind": "restricted", "hard": True,
             "polygon": [[10.66, 80.08], [10.66, 80.20], [10.82, 80.20], [10.82, 80.08]]},
            {"id": "GF-IMBL", "name": "International boundary buffer (illustrative)", "kind": "boundary", "hard": True,
             "polygon": [[9.95, 80.05], [9.95, 80.70], [10.25, 80.70], [10.25, 80.05]]},
            {"id": "GF-MPA", "name": "Point Calimere sanctuary waters (illustrative)", "kind": "mpa", "hard": True,
             "polygon": [[10.20, 79.85], [10.20, 80.00], [10.32, 80.00], [10.32, 79.85]]},
            {"id": "GF-PORT", "name": "Nagapattinam port approach channel (illustrative)", "kind": "port", "hard": False,
             "polygon": [[10.75, 79.84], [10.75, 79.92], [10.78, 79.92], [10.78, 79.84]]},
        ],
    }


def coast_doc():
    return {
        "meta": {"kind": "coast", "origin": "ILLUSTRATIVE simplified coastline polygon (land), not survey data"},
        "land": [[11.60, 79.00], [11.60, 79.845], [11.30, 79.850], [11.00, 79.855], [10.92, 79.850], [10.85, 79.850],
                 [10.77, 79.845], [10.60, 79.855], [10.45, 79.860], [10.30, 79.880], [10.15, 79.900], [9.90, 79.930],
                 [9.90, 79.00]],
    }


def gazetteer_doc():
    return {
        "ports": [
            {"id": "nagapattinam", "en": "Nagapattinam", "ta": ["நாகப்பட்டினம்", "நாகை"], "lat": 10.7672, "lon": 79.8449, "sea_lat": 10.7650, "sea_lon": 79.8700, "aliases": ["nagai", "negapatam"]},
            {"id": "karaikal", "en": "Karaikal", "ta": ["காரைக்கால்"], "lat": 10.9254, "lon": 79.8380, "sea_lat": 10.9254, "sea_lon": 79.8600, "aliases": []},
            {"id": "cuddalore", "en": "Cuddalore", "ta": ["கடலூர்"], "lat": 11.7480, "lon": 79.7714, "sea_lat": 11.7480, "sea_lon": 79.7950, "aliases": []},
            {"id": "puducherry", "en": "Puducherry", "ta": ["புதுச்சேரி", "பாண்டிச்சேரி"], "lat": 11.9416, "lon": 79.8083, "sea_lat": 11.9416, "sea_lon": 79.8300, "aliases": ["pondicherry"]},
            {"id": "chennai", "en": "Chennai", "ta": ["சென்னை"], "lat": 13.0827, "lon": 80.2707, "sea_lat": 13.0827, "sea_lon": 80.3000, "aliases": ["madras"]},
            {"id": "rameswaram", "en": "Rameswaram", "ta": ["ராமேஸ்வரம்", "இராமேசுவரம்"], "lat": 9.2876, "lon": 79.3129, "sea_lat": 9.2876, "sea_lon": 79.3400, "aliases": []},
            {"id": "thoothukudi", "en": "Thoothukudi", "ta": ["தூத்துக்குடி"], "lat": 8.7642, "lon": 78.1348, "sea_lat": 8.7642, "sea_lon": 78.1600, "aliases": ["tuticorin"]},
            {"id": "kanyakumari", "en": "Kanyakumari", "ta": ["கன்னியாகுமரி"], "lat": 8.0883, "lon": 77.5385, "sea_lat": 8.0883, "sea_lon": 77.5600, "aliases": ["cape comorin"]},
        ],
        "note": "Coordinates are approximate. Only the Nagapattinam area has bundled DEMO coverage.",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    docs = {"weather": weather_doc(), "ocean": ocean_doc(), "pfz": pfz_doc(), "advisories": advisory_doc(),
            "geofences": geofence_doc(), "coast": coast_doc(), "gazetteer": gazetteer_doc()}
    for name, doc in docs.items():
        p = OUT / f"{name}.json"
        p.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"{p.relative_to(OUT.parent.parent.parent)}  {p.stat().st_size:>7} bytes")


if __name__ == "__main__":
    main()
