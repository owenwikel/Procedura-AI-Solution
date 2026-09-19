#!/usr/bin/env python
"""Height-source coverage check for ML_ADDENDUM C.1.

For each of 20 Virginia Tech buildings: geocode (Nominatim), fetch the OSM
footprint (Overpass, SPEC 3.2 query + selection rule), and report whether an
authoritative height source exists (`height` or `building:levels` tag).

All HTTP responses are cached as JSON under cache/ and are never re-queried.
Missing geocodes / footprints are failures, reported loudly (exit code 1);
nothing is defaulted or synthesised.

Usage: python scripts/height_coverage_check.py
"""
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"

USER_AGENT = os.environ.get(
    "PROCEDURA_USER_AGENT", "procedura-height-coverage-check/0.1 (Virginia Tech hackathon)"
)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

NOMINATIM_MIN_INTERVAL_S = 1.1  # Nominatim policy: max 1 req/sec
OVERPASS_MIN_INTERVAL_S = 2.0
OVERPASS_MAX_ATTEMPTS = 6
OVERPASS_BACKOFF_BASE_S = 5.0
OVERPASS_BACKOFF_MAX_S = 120.0
OVERPASS_RETRY_STATUS = {429, 502, 503, 504}

SEARCH_RADIUS_M = 60  # SPEC 3.2 Overpass query
CANDIDATE_RADIUS_M = 50  # SPEC 3.2 selection rule, step 2
MIN_PLAUSIBLE_AREA_M2 = 50.0
COVERAGE_THRESHOLD = 0.80  # ML_ADDENDUM C.1

# Blacksburg sanity box: a geocode outside it is a wrong match, not a result.
BLACKSBURG_BBOX = (37.15, 37.30, -80.50, -80.35)  # lat_min, lat_max, lon_min, lon_max

EARTH_RADIUS_M = 6_371_000.0
NAME_TAGS = ("name", "official_name", "alt_name", "short_name")

# Geocoded as "<name>, Blacksburg, Virginia". Name queries rather than street numbers:
# OSM names campus buildings reliably, whereas a street address resolves to the
# generic 'Virginia Tech' point (800 Drillfield Dr is ~700 m from Burruss Hall).
# Do not add 'Virginia Tech' as a query component - Nominatim then returns nothing.
BUILDING_NAMES = [
    "Burruss Hall",
    "Goodwin Hall",
    "New Classroom Building",
    "Torgersen Hall",
    "Newman Library",
    "Squires Student Center",
    "McBryde Hall",
    "Holden Hall",
    "Norris Hall",
    "Randolph Hall",
    "Whittemore Hall",
    "Pamplin Hall",
    "Derring Hall",
    "Davidson Hall",
    "Hahn Hall",
    "Durham Hall",
    "War Memorial Hall",
    "Cassell Coliseum",
    "Lane Stadium",
    "Litton-Reaves Hall",
]
BUILDINGS = [(n, f"{n}, Blacksburg, Virginia") for n in BUILDING_NAMES]
assert len(BUILDINGS) == 20


class CheckError(Exception):
    """A building could not be resolved. Reported loudly; never defaulted."""


class GeocodeError(CheckError):
    pass


class FootprintError(CheckError):
    pass


class OverpassError(CheckError):
    pass


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------- cache


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def cache_path(kind, label, key):
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return CACHE_DIR / kind / f"{slugify(label)}_{digest}.json"


def cache_read(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["response"]
    return None


def cache_write(path, request, response):
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "request": request,
        "response": response,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # only complete responses ever appear in the cache


# ---------------------------------------------------------------- HTTP layer

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
_last_request_at = {}


def throttle(service, min_interval_s):
    wait = _last_request_at.get(service, 0.0) + min_interval_s - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_request_at[service] = time.monotonic()


def geocode(name, query):
    path = cache_path("nominatim", name, query)
    results = cache_read(path)
    if results is None:
        params = {"q": query, "format": "jsonv2", "limit": 1}
        throttle("nominatim", NOMINATIM_MIN_INTERVAL_S)
        resp = _session.get(NOMINATIM_URL, params=params, timeout=30)
        if resp.status_code != 200:
            raise GeocodeError(f"Nominatim HTTP {resp.status_code} for {query!r}: {resp.text[:200]}")
        results = resp.json()
        cache_write(path, {"url": NOMINATIM_URL, "params": params}, results)
    if not results:
        raise GeocodeError(f"Nominatim returned no result for {query!r}")
    hit = results[0]
    lat, lon = float(hit["lat"]), float(hit["lon"])
    lat_min, lat_max, lon_min, lon_max = BLACKSBURG_BBOX
    if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
        raise GeocodeError(
            f"Nominatim put {query!r} at ({lat:.5f}, {lon:.5f}) - outside Blacksburg: {hit.get('display_name')}"
        )
    return {
        "lat": lat,
        "lon": lon,
        "display_name": hit.get("display_name"),
        "osm_class": hit.get("category", hit.get("class")),
        "osm_type": hit.get("type"),
    }


def overpass_query(lat, lon):
    return (
        "[out:json][timeout:25];\n"
        "(\n"
        f'  way["building"](around:{SEARCH_RADIUS_M}, {lat:.6f}, {lon:.6f});\n'
        f'  relation["building"](around:{SEARCH_RADIUS_M}, {lat:.6f}, {lon:.6f});\n'
        ");\n"
        "out geom;"
    )


def overpass_fetch(query):
    """POST to Overpass, retrying 429/5xx/timeouts/runtime-error remarks with backoff."""
    last_problem = "no attempts made"
    for attempt in range(1, OVERPASS_MAX_ATTEMPTS + 1):
        throttle("overpass", OVERPASS_MIN_INTERVAL_S)
        retry_after = 0.0
        try:
            resp = _session.post(OVERPASS_URL, data={"data": query}, timeout=60)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_problem = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code == 200:
                try:
                    payload = resp.json()
                except ValueError:
                    raise OverpassError(f"Overpass returned non-JSON 200: {resp.text[:200]!r}")
                remark = payload.get("remark", "")
                if "runtime error" not in remark:
                    return payload
                last_problem = f"Overpass remark: {remark}"  # server-side timeout/OOM
            elif resp.status_code in OVERPASS_RETRY_STATUS:
                last_problem = f"HTTP {resp.status_code}"
                header = resp.headers.get("Retry-After", "")
                retry_after = float(header) if header.isdigit() else 0.0
            else:
                raise OverpassError(f"Overpass HTTP {resp.status_code}: {resp.text[:200]!r}")
        if attempt == OVERPASS_MAX_ATTEMPTS:
            break
        delay = min(max(OVERPASS_BACKOFF_BASE_S * 2 ** (attempt - 1), retry_after), OVERPASS_BACKOFF_MAX_S)
        log(f"    overpass attempt {attempt}/{OVERPASS_MAX_ATTEMPTS} failed ({last_problem}); retrying in {delay:.0f}s")
        time.sleep(delay)
    raise OverpassError(f"Overpass failed after {OVERPASS_MAX_ATTEMPTS} attempts: {last_problem}")


def fetch_overpass(name, lat, lon):
    query = overpass_query(lat, lon)
    path = cache_path("overpass", name, query)
    payload = cache_read(path)
    if payload is None:
        payload = overpass_fetch(query)
        cache_write(path, {"url": OVERPASS_URL, "query": query}, payload)
    return payload


# ------------------------------------------------------------------ geometry
# Everything is projected to a local tangent plane (metres) centred on the
# geocoded point, so the geocoded point is the origin. Fine at building scale.


def make_projector(lat0, lon0):
    kx = math.radians(1.0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    ky = math.radians(1.0) * EARTH_RADIUS_M
    return lambda lon, lat: ((lon - lon0) * kx, (lat - lat0) * ky)


def ring_area(ring):
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(ring, ring[1:]))) / 2.0


def ring_centroid(ring):
    a = cx = cy = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        cross = x1 * y2 - x2 * y1
        a += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if a == 0:
        raise FootprintError("degenerate ring (zero area)")
    return cx / (3 * a), cy / (3 * a)


def point_in_ring(x, y, ring):
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def convex_hull(points):
    """Andrew's monotone chain; returns a closed ring (first point repeated)."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and (
                (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1]) - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0]) <= 0
            ):
                out.pop()
            out.append(p)
        return out

    hull = half(pts)[:-1] + half(reversed(pts))[:-1]
    return hull + [hull[0]]


def dist_to_ring(x, y, ring):
    best = math.inf
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        dx, dy = x2 - x1, y2 - y1
        t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)))
        best = min(best, math.hypot(x - (x1 + t * dx), y - (y1 + t * dy)))
    return best


def stitch_rings(segments):
    """Join OSM way fragments (lists of (lon, lat)) into closed rings."""
    segments = [list(s) for s in segments if len(s) >= 2]
    rings = []
    while segments:
        ring = segments.pop()
        while ring[0] != ring[-1]:
            for i, seg in enumerate(segments):
                if seg[0] == ring[-1]:
                    ring.extend(seg[1:])
                elif seg[-1] == ring[-1]:
                    ring.extend(reversed(seg[:-1]))
                elif seg[-1] == ring[0]:
                    ring[:0] = seg[:-1]
                elif seg[0] == ring[0]:
                    ring[:0] = list(reversed(seg))[:-1]
                else:
                    continue
                del segments[i]
                break
            else:
                raise FootprintError("multipolygon ring does not close")
        if len(ring) < 4:
            raise FootprintError("degenerate ring (<3 vertices)")
        rings.append(ring)
    return rings


def element_rings(element):
    """Return (outer_rings, inner_rings) as lists of (lon, lat) rings."""
    def coords(geometry):
        return [(p["lon"], p["lat"]) for p in geometry]

    if element["type"] == "way":
        ring = coords(element.get("geometry", []))
        if len(ring) < 4 or ring[0] != ring[-1]:
            raise FootprintError("way is not a closed ring")
        return [ring], []
    outer, inner = [], []
    for member in element.get("members", []):
        if member["type"] != "way" or "geometry" not in member:
            continue
        (inner if member.get("role") == "inner" else outer).append(coords(member["geometry"]))
    if not outer:
        raise FootprintError("relation has no outer way geometry")
    return stitch_rings(outer), stitch_rings(inner)


def build_candidate(element, project):
    outer_ll, inner_ll = element_rings(element)
    outers = [[project(*p) for p in ring] for ring in outer_ll]
    inners = [[project(*p) for p in ring] for ring in inner_ll]
    # Holes count against area (SPEC 3.3).
    area = sum(map(ring_area, outers)) - sum(map(ring_area, inners))
    largest = max(outers, key=ring_area)
    in_hole = any(point_in_ring(0.0, 0.0, i) for i in inners)
    in_outer = any(point_in_ring(0.0, 0.0, o) for o in outers)
    # A multi-polygon relation is one building split into several outer rings
    # (e.g. a stadium's four stands around a field). A point in the gap between
    # them is inside the relation, so test the outer rings collectively via
    # their convex hull. Holes still exclude.
    in_hull = (
        len(outers) > 1
        and not in_outer
        and point_in_ring(0.0, 0.0, convex_hull([p for o in outers for p in o]))
    )
    contains = (in_outer or in_hull) and not in_hole
    cx, cy = ring_centroid(largest)
    return {
        "id": f"{element['type']}/{element['id']}",
        "tags": element.get("tags", {}),
        "area_m2": area,
        "contains_point": contains,
        "contained_via": "ring" if in_outer else "relation_hull",
        "boundary_dist_m": 0.0 if contains else min(dist_to_ring(0.0, 0.0, o) for o in outers),
        "centroid_dist_m": math.hypot(cx, cy),
    }


def norm_name(text):
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def name_matches(target, tags):
    target = norm_name(target)
    for key in NAME_TAGS:
        for value in tags.get(key, "").split(";"):
            value = norm_name(value)
            # OSM name must contain the full query name ('Hahn Hall South' matches
            # 'Hahn Hall'); the reverse would let a tag like 'Hall' match anything.
            if value and target in value:
                return True
    return False


def select_footprint(name, payload, project):
    """SPEC 3.2 selection rule. Raises FootprintError rather than guessing."""
    candidates, skipped = [], []
    for element in payload.get("elements", []):
        if element.get("type") not in ("way", "relation"):
            continue
        try:
            candidates.append(build_candidate(element, project))
        except FootprintError as exc:
            skipped.append(f"{element['type']}/{element['id']}: {exc}")
    for msg in skipped:
        log(f"    WARNING skipped unusable building geometry {msg}")

    hits = [c for c in candidates if c["contains_point"]]
    if len(hits) == 1:
        method = "point_in_polygon" if hits[0]["contained_via"] == "ring" else "point_in_relation_hull"
        return hits[0], method, 1
    pool = hits or [c for c in candidates if c["boundary_dist_m"] <= CANDIDATE_RADIUS_M]
    if not pool:
        raise FootprintError(
            f"no building footprint within {CANDIDATE_RADIUS_M} m of the geocoded point "
            f"({len(candidates)} candidates within {SEARCH_RADIUS_M} m, {len(skipped)} unusable)"
        )
    pool.sort(
        key=lambda c: (
            not name_matches(name, c["tags"]),
            c["centroid_dist_m"],
            c["area_m2"] < MIN_PLAUSIBLE_AREA_M2,
        )
    )
    best = pool[0]
    # Ties (several containing polygons) and nearby picks both need name evidence;
    # centroid distance alone is a guess.
    if name_matches(name, best["tags"]):
        return best, "pip_ranked" if hits else "nearby_ranked", len(pool)
    if len(pool) == 1 and not any(best["tags"].get(k) for k in NAME_TAGS):
        return best, "unnamed_sole_candidate", 1
    found = ", ".join(f"{c['id']} {c['tags'].get('name')!r} ({c['boundary_dist_m']:.0f} m)" for c in pool)
    where = "containing the point" if hits else f"within {CANDIDATE_RADIUS_M} m"
    raise FootprintError(
        f"no name match for {name!r} among {len(pool)} building(s) {where} "
        f"(picking by distance alone is a guess, refusing): {found}"
    )


# -------------------------------------------------------------- height tags


def parse_height_m(raw):
    """OSM `height`: metres by default; accepts '25', '25.5 m', '82 ft', "82'". None if unparseable."""
    match = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*(m|ft|')?\s*", raw or "")
    if not match:
        return None
    value = float(match.group(1).replace(",", "."))
    return value * 0.3048 if match.group(2) in ("ft", "'") else value


def parse_levels(raw):
    match = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*", raw or "")
    return float(match.group(1).replace(",", ".")) if match else None


def analyze(name, query):
    place = geocode(name, query)
    payload = fetch_overpass(name, place["lat"], place["lon"])
    project = make_projector(place["lat"], place["lon"])
    chosen, method, pool_size = select_footprint(name, payload, project)
    tags = chosen["tags"]
    raw_height, raw_levels = tags.get("height"), tags.get("building:levels")
    height_m, levels = parse_height_m(raw_height), parse_levels(raw_levels)
    return {
        "name": name,
        "footprint_id": chosen["id"],
        "osm_name": tags.get("name"),
        "selection": method,
        "pool_size": pool_size,
        "name_match": name_matches(name, tags),
        "area_m2": chosen["area_m2"],
        "height_tag": raw_height,
        "height_m": height_m,
        "levels_tag": raw_levels,
        "levels": levels,
        "authoritative": height_m is not None or levels is not None,
    }


# -------------------------------------------------------------------- report


def yes_no(tag, parsed):
    if tag is None:
        return "no"
    return f"yes ({tag})" if parsed is not None else f"UNPARSEABLE ({tag})"


def main():
    rows, failures = [], []
    for i, (name, query) in enumerate(BUILDINGS, 1):
        log(f"[{i:2d}/{len(BUILDINGS)}] {name}")
        try:
            rows.append(analyze(name, query))
        except CheckError as exc:
            log(f"    FAILED: {exc}")
            failures.append((name, f"{type(exc).__name__}: {exc}"))

    print(f"\n{'building':<24} {'footprint':<16} {'select':<22} {'area m2':>8}  {'height tag':<18} {'building:levels':<18} auth")
    print("-" * 116)
    for r in rows:
        flags = []
        if r["pool_size"] > 1:
            flags.append(f"AMBIGUOUS({r['pool_size']})")
        if not r["name_match"]:
            flags.append(f"NAME-MISMATCH(osm={r['osm_name']!r})")
        print(
            f"{r['name']:<24} {r['footprint_id']:<16} {r['selection']:<22} {r['area_m2']:>8.0f}  "
            f"{yes_no(r['height_tag'], r['height_m']):<18} {yes_no(r['levels_tag'], r['levels']):<18} "
            f"{'YES' if r['authoritative'] else 'no':<4} {' '.join(flags)}"
        )
    for name, why in failures:
        print(f"{name:<24} *** FAILED *** {why}")

    total = len(BUILDINGS)
    covered = sum(r["authoritative"] for r in rows)
    with_height = sum(r["height_m"] is not None for r in rows)
    with_levels = sum(r["levels"] is not None for r in rows)
    print(f"\nAuthoritative height source (OSM height or building:levels): {covered} of {total}")
    print(f"  height tag: {with_height}   building:levels: {with_levels}   footprints resolved: {len(rows)} of {total}")
    if failures:
        print(f"\n{len(failures)} building(s) FAILED - the coverage figure above is a lower bound, not a result.")
        return 1
    verdict = "SKIP ML_ADDENDUM section C" if covered / total >= COVERAGE_THRESHOLD else "BUILD ML_ADDENDUM section C"
    print(f"C.1 threshold {COVERAGE_THRESHOLD:.0%} ({math.ceil(COVERAGE_THRESHOLD * total)} of {total}): {covered / total:.0%} -> {verdict}")
    print("Note: OSM only; C.1 also asks for a Microsoft Building Footprints height check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
