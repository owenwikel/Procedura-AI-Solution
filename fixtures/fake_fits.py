"""Synthetic labelled placements for building/testing the confidence model (ML_ADDENDUM F.4).

    rows = generate(seed=0)          # 120 rows by default (B.2: 80 benchmark + 40 corruptions)
    row.fit, row.photo, row.footprint, row.geocode_rooftop, row.height_source_authoritative
    row.label                        # 1 = a human would accept this placement into the game world

The point is to exercise features.py and train.py before the solver has produced real benchmark
runs. NUMBERS TRAINED ON THIS DATA DEMONSTRATE THE PIPELINE, NOT THE PIPELINE'S ACCURACY: the
signal is whatever this generator puts in.

How the labels avoid being contaminated by the features (B.4): the label is drawn FIRST, from
the building's difficulty and the ablation config's quality, and the metrics are then sampled
conditional on it. Bad placements come in overlapping flavours (a "borderline" one looks a lot
like a good one), and a few labels are flipped to mimic annotator disagreement, so the classes
are not cleanly separable.

Rows follow B.2: `n_buildings` buildings (cycling the four strata: rectangular, complex,
near-square, sloped) x `n_configs` ablation configs of rising solver quality, plus
`n_corruptions` deliberate corruptions (a known-wrong orientation, a mirrored mesh, a 2x scale
error), which are always labelled 0.

contracts.py is frozen and shared; if it is importable its FitResult / PhotoEvidence / Footprint are
used. Until it lands on this branch the fallback classes below mirror ML_ADDENDUM F.1 exactly.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from contracts import FitResult, Footprint, PhotoEvidence
except ImportError:  # contracts.py has not landed on this branch: mirror ML_ADDENDUM F.1

    @dataclass(frozen=True)
    class Footprint:
        pts_enu: np.ndarray
        rectilinearity: float
        ombb: tuple
        has_holes: bool

    @dataclass(frozen=True)
    class PhotoEvidence:
        mask: np.ndarray
        mask_area_frac: float
        occlusion_flag: bool
        exif_heading_deg: float | None
        exif_pitch_deg: float | None
        silhouette_scores: dict
        silhouette_margin: float

    @dataclass(frozen=True)
    class FitResult:
        theta: float
        scale_x: float
        scale_y: float
        tx: float
        ty: float
        z_offset: float
        iou: float
        hausdorff_m: float
        area_ratio: float
        rotation_margin_footprint: float
        anisotropy_log_ratio: float
        max_neighbor_overlap: float
        disambiguated_by: str


@dataclass(frozen=True)
class LabelledFit:
    fit: FitResult
    photo: PhotoEvidence
    footprint: Footprint
    geocode_rooftop: bool  # SPEC 3.1 (not in the three contract objects)
    height_source_authoritative: bool  # SPEC 7 (likewise)
    label: int  # 1 = acceptable, 0 = not
    kind: str  # what the generator drew before label noise
    building_id: int
    config_index: int | None  # ablation config, None for a deliberate corruption
    label_flipped: bool


STRATA = ("rectangular", "complex", "near_square", "sloped")
CONFIG_QUALITY = (0.0, 0.35, 0.7, 0.9)  # ablation: OMBB only, +Umeyama, +IoU refinement, +silhouette
CORRUPTIONS = ("corrupt_wrong_orientation", "corrupt_mirrored", "corrupt_scale_2x")
BAD_BENCHMARK_KINDS = (("poor_fit", 0.35), ("wrong_orientation", 0.35), ("borderline", 0.30))


def _clip(x, lo, hi):
    return float(min(max(x, lo), hi))


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _lognormal(rng, median, sigma):
    return float(rng.lognormal(math.log(median), sigma))


@dataclass(frozen=True)
class _Building:
    building_id: int
    stratum: str
    rectilinearity: float
    aspect: float
    area_m2: float
    has_holes: bool
    geocode_rooftop: bool
    height_authoritative: bool
    rot_margin_base: float  # how decisive the footprint's four OMBB candidates are
    sil_margin_base: float  # how decisive the photo silhouette is


def _make_building(rng, building_id):
    stratum = STRATA[building_id % len(STRATA)]
    r_lo, r_hi = {"rectangular": (0.92, 0.99), "complex": (0.72, 0.90), "near_square": (0.90, 0.98), "sloped": (0.85, 0.97)}[stratum]
    near_square = stratum == "near_square"
    return _Building(
        building_id=building_id,
        stratum=stratum,
        rectilinearity=float(rng.uniform(r_lo, r_hi)),
        aspect=float(rng.uniform(1.0, 1.09) if near_square else rng.uniform(1.3, 3.0)),
        area_m2=float(rng.uniform(400, 4000)),
        has_holes=bool(rng.random() < (0.3 if stratum == "complex" else 0.05)),
        geocode_rooftop=bool(rng.random() < 0.75),
        height_authoritative=bool(rng.random() < 0.65),
        rot_margin_base=float(rng.uniform(0.0, 0.05) if near_square else rng.uniform(0.12, 0.5)),
        sil_margin_base=float(rng.uniform(0.0, 0.04) if near_square else rng.uniform(0.05, 0.35)),
    )


def _p_acceptable(b, quality):
    """Chance a placement of this building at this solver quality is one a human accepts."""
    z = -1.4 + 3.2 * quality + 5.0 * (b.rectilinearity - 0.9) + 0.5 * b.geocode_rooftop
    z += 0.3 * b.height_authoritative - 0.8 * (b.stratum == "near_square")
    return _sigmoid(z)


def _metrics(rng, kind, b):
    """Metric draws for one placement of `kind` on building `b` (signed log quantities)."""
    c = 0.95 - b.rectilinearity  # complexity: hurts even good fits a little
    sign = 1.0 if rng.random() < 0.5 else -1.0
    rot, sil = b.rot_margin_base, b.sil_margin_base
    if kind == "good":
        m = dict(iou=rng.normal(0.87 - 0.5 * c, 0.045), haus=_lognormal(rng, 1.5 + 4 * c, 0.3), arl=rng.normal(0, 0.05),
                 aniso=rng.normal(0, 0.035), overlap=rng.beta(1, 80), rot=rot * rng.uniform(0.8, 1.2), sil=sil * rng.uniform(0.7, 1.3))
    elif kind == "borderline":
        m = dict(iou=rng.normal(0.79, 0.05), haus=_lognormal(rng, 2.6, 0.3), arl=rng.normal(0, 0.10),
                 aniso=rng.normal(0, 0.07), overlap=rng.beta(1, 40), rot=rot * rng.uniform(0.5, 1.0), sil=sil * rng.uniform(0.4, 1.0))
    elif kind == "poor_fit":
        m = dict(iou=rng.normal(0.68, 0.07), haus=_lognormal(rng, 3.6, 0.3), arl=sign * rng.normal(0.28, 0.10),
                 aniso=rng.normal(0, 0.14), overlap=rng.beta(1, 15), rot=rot * rng.uniform(0.3, 0.9), sil=sil * rng.uniform(0.2, 0.8))
    elif kind == "wrong_orientation":
        m = dict(iou=rng.normal(0.60, 0.08), haus=_lognormal(rng, 4.5, 0.3), arl=rng.normal(0, 0.15),
                 aniso=rng.normal(0, 0.08), overlap=rng.beta(1, 25), rot=rng.uniform(0, 0.05), sil=rng.uniform(0, 0.05))
    elif kind == "corrupt_wrong_orientation":
        m = dict(iou=rng.normal(0.50, 0.08), haus=_lognormal(rng, 6.0, 0.3), arl=rng.normal(0, 0.10),
                 aniso=rng.normal(0, 0.05), overlap=rng.beta(1, 20), rot=rng.uniform(0, 0.06), sil=rng.uniform(0, 0.06))
    elif kind == "corrupt_mirrored":
        m = dict(iou=rng.normal(0.45, 0.09), haus=_lognormal(rng, 7.0, 0.3), arl=rng.normal(0, 0.12),
                 aniso=rng.normal(0, 0.06), overlap=rng.beta(1, 15), rot=rng.uniform(0, 0.10), sil=rng.uniform(0, 0.10))
    elif kind == "corrupt_scale_2x":
        m = dict(iou=rng.normal(0.32, 0.07), haus=_lognormal(rng, 8.0, 0.3), arl=sign * (math.log(2) + rng.normal(0, 0.05)),
                 aniso=rng.normal(0, 0.03), overlap=rng.beta(1, 8), rot=rot, sil=sil)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return dict(
        iou=_clip(m["iou"], 0.05, 0.99), haus=max(0.05, m["haus"]), arl=float(m["arl"]), aniso=float(m["aniso"]),
        overlap=_clip(m["overlap"], 0.0, 1.0), rot=_clip(m["rot"], 0.0, 1.0), sil=_clip(m["sil"], 0.0, 1.0),
    )


def _disambiguated_by(rng, kind):
    probs = {"good": (0.25, 0.55, 0.20), "borderline": (0.2, 0.4, 0.4), "poor_fit": (0.2, 0.4, 0.4)}.get(kind, (0.1, 0.3, 0.6))
    return str(rng.choice(["exif_heading", "silhouette", "road_normal"], p=probs))


def _build_row(rng, b, kind, label, config_index, label_flipped):
    m = _metrics(rng, kind, b)
    # Scales coherent with the metrics: area_ratio = s_x * s_y and anisotropy = log(s_x / s_y).
    log_sx, log_sy = m["arl"] / 2 + m["aniso"] / 2, m["arl"] / 2 - m["aniso"] / 2
    fit = FitResult(
        theta=float(rng.uniform(0, 2 * math.pi)), scale_x=math.exp(log_sx), scale_y=math.exp(log_sy),
        tx=float(rng.normal(0, 0.8)), ty=float(rng.normal(0, 0.8)), z_offset=float(rng.uniform(600, 660)),
        iou=m["iou"], hausdorff_m=m["haus"], area_ratio=math.exp(m["arl"]), rotation_margin_footprint=m["rot"],
        anisotropy_log_ratio=m["aniso"], max_neighbor_overlap=m["overlap"], disambiguated_by=_disambiguated_by(rng, kind),
    )
    # Hardcoded-style mask: a centred rectangle covering roughly a third to two thirds of the frame.
    side = int(round(32 * math.sqrt(rng.uniform(0.25, 0.7))))
    mask = np.zeros((32, 32), bool)
    lo = (32 - side) // 2
    mask[lo : lo + side, lo : lo + side] = True
    best = float(rng.uniform(0.65, 0.95))
    second = max(0.0, best - m["sil"])
    scores = {(0, k): max(0.0, second - float(rng.uniform(0, 0.2))) for k in range(4)}
    best_k, second_k = rng.choice(4, size=2, replace=False)
    scores[(0, int(best_k))], scores[(0, int(second_k))] = best, second
    has_exif = rng.random() < 0.4
    photo = PhotoEvidence(
        mask=mask, mask_area_frac=float(mask.mean()), occlusion_flag=bool(rng.random() < 0.1),
        exif_heading_deg=float(rng.uniform(0, 360)) if has_exif else None,
        exif_pitch_deg=float(rng.normal(8, 6)) if has_exif or rng.random() < 0.4 else None,
        silhouette_scores=scores, silhouette_margin=best - second,
    )
    short = math.sqrt(b.area_m2 / b.aspect)
    a, s = b.aspect * short, short
    pts = np.array([[-a / 2, -s / 2], [a / 2, -s / 2], [a / 2, s / 2], [-a / 2, s / 2]])
    footprint = Footprint(pts_enu=pts, rectilinearity=b.rectilinearity, ombb=((0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)), (a, s)), has_holes=b.has_holes)
    return LabelledFit(fit, photo, footprint, b.geocode_rooftop, b.height_authoritative, int(label), kind, b.building_id, config_index, label_flipped)


def generate(n_buildings=20, n_configs=4, n_corruptions=40, seed=0, label_noise=0.05):
    """Deterministic for a given seed. Returns n_buildings * n_configs + n_corruptions LabelledFit rows."""
    if not 0 <= label_noise < 0.5:
        raise ValueError("label_noise must be in [0, 0.5)")
    rng = np.random.default_rng(seed)
    buildings = [_make_building(rng, i) for i in range(n_buildings)]
    rows = []
    for b in buildings:
        for c in range(n_configs):
            acceptable = rng.random() < _p_acceptable(b, CONFIG_QUALITY[c % len(CONFIG_QUALITY)])
            if acceptable:
                kind = "good"
            else:
                names, probs = zip(*BAD_BENCHMARK_KINDS)
                kind = str(rng.choice(names, p=probs))
            flipped = bool(rng.random() < label_noise)  # annotator disagreement
            rows.append(_build_row(rng, b, kind, int(acceptable) ^ int(flipped), c, flipped))
    for i in range(n_corruptions):
        b = buildings[i % n_buildings]
        rows.append(_build_row(rng, b, CORRUPTIONS[i % len(CORRUPTIONS)], 0, None, False))
    return rows


if __name__ == "__main__":
    rows = generate()
    kinds = {}
    for r in rows:
        kinds.setdefault(r.kind, [0, 0])[r.label] += 1
    print(f"{len(rows)} rows, base rate of acceptable = {np.mean([r.label for r in rows]):.2f}")
    for kind, (neg, pos) in sorted(kinds.items()):
        print(f"  {kind:<26} label0={neg:3d} label1={pos:3d}")
