#!/usr/bin/env python
"""Orientation fusion ladder (ML_ADDENDUM A.3 "Fusion with the other filters", SPEC 6.6).

    choice = choose_orientation(photo, comparison, facade_heading, road_normal_deg=None)
    choice.candidate, choice.disambiguated_by, choice.needs_review, choice.reason

Precedence (never let render-and-compare silently override a confident EXIF heading):
  1. EXIF GPSImgDirection present and plausible  -> it decides      "exif_heading"
  2. else silhouette margin >= 0.08              -> silhouette      "silhouette"
  3. else road-normal prior (SPEC 6.6 Filter 3)  -> nearest facade  "road_normal", needs_review
     regardless of IoU: you are now guessing.
  Disagreement is recorded, never acted on: `exif_silhouette_disagree` is True when EXIF decided AND
  the silhouette margin was >= 0.08 AND the silhouette's best candidate differs from EXIF's pick.
  EXIF still wins; the flag (with `silhouette_margin`, recorded on every branch) is a feature for
  the confidence gate, since the two are independent evidence and their conflict signals a problem.
  If none of these has evidence, the silhouette's best candidate is returned, flagged for
  review, as "arbitrary_symmetric" when Comparison.label says so (A.4), else "unresolved".
  "unresolved" is not in the spec's vocabulary; map it as you see fit downstream.

Inputs
  photo        PhotoEvidence-shaped: only `exif_heading_deg` (float | None) is read. contracts.py
               is frozen, so this is duck-typed rather than imported. `photo.silhouette_margin`
               is ignored: `comparison.margin` is the source of truth.
  comparison   a render_compare.Comparison (needs .candidates, .best, .margin, .label).
  facade_heading  callable(Candidate) -> float, supplied by geo/. The compass bearing (degrees
               clockwise from north) of the outward normal of the facade the photo shows, if that
               candidate were the true orientation. Perception cannot compute it: it depends on the
               footprint rotation theta_k = theta0 + k*90 deg (SPEC 6.5). SPEC 2.4: bearing =
               90 deg - theta for a math angle theta. Required whenever EXIF or the road normal is
               used; a missing mapping raises rather than silently skipping evidence.
  road_normal_deg  bearing (degrees clockwise from north) of the footprint's outward normal toward
               the nearest road. Fetched by geo/, never here. `bearing_from_enu` converts (east, north).

Up-axis is not something EXIF or a road can resolve, so branches 1 and 3 pick among the four
azimuths that share the silhouette best's up-axis. The EXIF heading is the camera's pointing
direction, so the photographed facade faces the opposite bearing (heading + 180). EXIF is
assumed to be relative to true north (PhotoEvidence does not carry GPSImgDirectionRef); an
implausible value (outside [0, 360), non-finite, non-numeric) is ignored and the reason says so.
Ties (a heading or normal exactly between two facades) go to the better silhouette score.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

SILHOUETTE_DECISIVE_MARGIN = 0.08  # A.3 precedence rule 2
EXIF_HEADING = "exif_heading"
SILHOUETTE = "silhouette"
ROAD_NORMAL = "road_normal"
ARBITRARY_SYMMETRIC = "arbitrary_symmetric"  # matches render_compare.ARBITRARY_SYMMETRIC (A.4)
UNRESOLVED = "unresolved"  # no evidence at all; not in the spec's vocabulary
_TIE_DECIMALS = 6  # angular differences equal to this many decimals count as a tie


@dataclass(frozen=True)
class OrientationChoice:
    candidate: object  # a render_compare.Candidate
    disambiguated_by: str
    needs_review: bool
    reason: str  # human-readable audit trail for the intermediate-artifact log
    silhouette_margin: float  # comparison.margin, whichever branch decided (a confidence-gate feature)
    exif_silhouette_disagree: bool = False  # EXIF decided, silhouette was confident, and picked another candidate


def bearing_from_enu(east: float, north: float) -> float:
    """Compass bearing (degrees clockwise from north, [0, 360)) of an ENU vector."""
    return math.degrees(math.atan2(east, north)) % 360.0


def angular_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings, in [0, 180]."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def plausible_heading(value) -> Optional[float]:
    """EXIF GPSImgDirection as a float in [0, 360), else None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        heading = float(value)
    except (TypeError, ValueError):
        return None
    return heading if math.isfinite(heading) and 0.0 <= heading < 360.0 else None


def _nearest_facade(candidates, bearing, facade_heading):
    """(candidate, its facade bearing, angular difference) closest to `bearing`; ties -> better score."""
    ranked = []
    for c in candidates:
        facing = float(facade_heading(c))
        if not math.isfinite(facing):
            raise ValueError(f"facade_heading returned {facing!r} for candidate {c}")
        ranked.append((round(angular_diff_deg(facing, bearing), _TIE_DECIMALS), -c.score, c, facing))
    diff, _, chosen, facing = min(ranked, key=lambda r: r[:2])
    return chosen, facing, diff


def _need_mapping(facade_heading, evidence):
    if facade_heading is None:
        raise ValueError(f"{evidence} is available but facade_heading was not supplied; cannot map it to a candidate")


def choose_orientation(
    photo, comparison, facade_heading: Optional[Callable] = None, road_normal_deg: Optional[float] = None
) -> OrientationChoice:
    up = comparison.best.up_axis
    same_up = [c for c in comparison.candidates if c.up_axis == up]
    raw = photo.exif_heading_deg
    heading = plausible_heading(raw)
    note = f"exif heading {raw!r} implausible, ignored; " if raw is not None and heading is None else ""

    # 1. EXIF heading decides.
    if heading is not None:
        _need_mapping(facade_heading, "an EXIF heading")
        facing = (heading + 180.0) % 360.0  # the photographed facade faces back toward the camera
        chosen, bearing, diff = _nearest_facade(same_up, facing, facade_heading)
        best = comparison.best
        disagree = comparison.margin >= SILHOUETTE_DECISIVE_MARGIN and (chosen.up_axis, chosen.azimuth_deg) != (
            best.up_axis,
            best.azimuth_deg,
        )
        reason = (
            f"exif heading {heading:.1f} -> photographed facade faces {facing:.1f}; "
            f"nearest candidate faces {bearing:.1f} ({diff:.1f} deg off)"
        )
        if disagree:
            reason += f"; DISAGREES with a confident silhouette (margin {comparison.margin:.3f}, prefers azimuth {best.azimuth_deg:.0f})"
        return OrientationChoice(chosen, EXIF_HEADING, False, reason, comparison.margin, disagree)

    # 2. A confident silhouette decides.
    if comparison.margin >= SILHOUETTE_DECISIVE_MARGIN:
        return OrientationChoice(
            comparison.best, SILHOUETTE, False,
            f"{note}silhouette margin {comparison.margin:.3f} >= {SILHOUETTE_DECISIVE_MARGIN}",
            comparison.margin,
        )

    # 3. Road-normal prior; a guess, so always review.
    if road_normal_deg is not None:
        road = float(road_normal_deg)
        if not math.isfinite(road):
            raise ValueError(f"road_normal_deg must be finite, got {road_normal_deg!r}")
        _need_mapping(facade_heading, "a road normal")
        road %= 360.0
        chosen, bearing, diff = _nearest_facade(same_up, road, facade_heading)
        return OrientationChoice(
            chosen, ROAD_NORMAL, True,
            f"{note}silhouette margin {comparison.margin:.3f} < {SILHOUETTE_DECISIVE_MARGIN} and no usable EXIF; "
            f"road normal {road:.1f} -> nearest candidate faces {bearing:.1f} ({diff:.1f} deg off); guessing",
            comparison.margin,
        )

    # No evidence left: return the silhouette's pick, visibly unreliable.
    label = getattr(comparison, "label", None)
    return OrientationChoice(
        comparison.best, ARBITRARY_SYMMETRIC if label == ARBITRARY_SYMMETRIC else UNRESOLVED, True,
        f"{note}silhouette margin {comparison.margin:.3f} < {SILHOUETTE_DECISIVE_MARGIN}, no usable EXIF, "
        "no road normal; returning the silhouette's best candidate",
        comparison.margin,
    )
