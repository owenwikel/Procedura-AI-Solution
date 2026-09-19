"""Feature vector for the learned confidence gate (ML_ADDENDUM B.3).

    x = feature_vector(fit, photo, fp, geocode_rooftop=..., height_source_authoritative=...)

Inputs are duck-typed (contracts.py is frozen and shared): `fit` needs iou, hausdorff_m, area_ratio,
rotation_margin_footprint, anisotropy_log_ratio, max_neighbor_overlap; `photo` needs
silhouette_margin; `fp` needs rectilinearity. Nothing is defaulted: a missing attribute raises
AttributeError, and a non-finite value or a non-positive area ratio raises ValueError.

The two binary features are NOT in FitResult / PhotoEvidence / Footprint (F.1), so they are
required keyword arguments. `score_confidence(fit, photo, fp)` as written in F.1 has no way to
supply them; that signature needs a contract decision before gate.py can call this.

Feature table (B.3), with the expected sign the fitted coefficient is compared against:

  footprint_iou                +   fit.iou                                  SPEC 9.1
  hausdorff_m                  -   fit.hausdorff_m                          SPEC 9.1
  abs_area_ratio_log           -   |log(fit.area_ratio)|                    SPEC 9.1
  rotation_margin_footprint    +   fit.rotation_margin_footprint            SPEC 6.6
  rotation_margin_silhouette   +   photo.silhouette_margin                  A.3
  rectilinearity               +   fp.rectilinearity                        SPEC 4.2
  abs_anisotropy_log_ratio     -   |fit.anisotropy_log_ratio|               SPEC 6.9
  max_neighbor_overlap         -   fit.max_neighbor_overlap                 SPEC 9.2
  geocode_rooftop              +   caller (Google location_type == ROOFTOP) SPEC 3.1
  height_source_authoritative  +   caller (OSM height / levels tag)         SPEC 7

The table's `area_ratio_log` is used as |log(ratio)|: log makes it symmetric about 1 (0.5x and 2x
are equally wrong) and the "||.|| -" sign means the magnitude is what hurts, which a linear model
can only express on the absolute value. The same goes for the sign-arbitrary anisotropy log-ratio.
"""
from __future__ import annotations

import math

import numpy as np

FEATURE_NAMES = (
    "footprint_iou",
    "hausdorff_m",
    "abs_area_ratio_log",
    "rotation_margin_footprint",
    "rotation_margin_silhouette",
    "rectilinearity",
    "abs_anisotropy_log_ratio",
    "max_neighbor_overlap",
    "geocode_rooftop",
    "height_source_authoritative",
)

EXPECTED_SIGN = {
    "footprint_iou": +1,
    "hausdorff_m": -1,
    "abs_area_ratio_log": -1,
    "rotation_margin_footprint": +1,
    "rotation_margin_silhouette": +1,
    "rectilinearity": +1,
    "abs_anisotropy_log_ratio": -1,
    "max_neighbor_overlap": -1,
    "geocode_rooftop": +1,
    "height_source_authoritative": +1,
}

# B.4 "too few rows" response: 4 features on 60 rows is defensible, 10 is not.
REDUCED_FEATURES = ("footprint_iou", "hausdorff_m", "rectilinearity", "rotation_margin_footprint")

assert set(EXPECTED_SIGN) == set(FEATURE_NAMES) and set(REDUCED_FEATURES) <= set(FEATURE_NAMES)


def _finite(name, value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"feature {name} is not finite: {value!r}")
    return value


def _binary(name, value):
    if isinstance(value, (bool, np.bool_)) or (isinstance(value, (int, np.integer)) and value in (0, 1)):
        return float(bool(value))
    raise ValueError(f"feature {name} must be a bool (or 0/1), got {value!r}")


def feature_dict(fit, photo, fp, *, geocode_rooftop, height_source_authoritative):
    """All ten B.3 features, by name."""
    area_ratio = _finite("area_ratio", fit.area_ratio)
    if area_ratio <= 0:
        raise ValueError(f"fit.area_ratio must be positive to take its log, got {area_ratio!r}")
    return {
        "footprint_iou": _finite("footprint_iou", fit.iou),
        "hausdorff_m": _finite("hausdorff_m", fit.hausdorff_m),
        "abs_area_ratio_log": abs(math.log(area_ratio)),
        "rotation_margin_footprint": _finite("rotation_margin_footprint", fit.rotation_margin_footprint),
        "rotation_margin_silhouette": _finite("rotation_margin_silhouette", photo.silhouette_margin),
        "rectilinearity": _finite("rectilinearity", fp.rectilinearity),
        "abs_anisotropy_log_ratio": abs(_finite("anisotropy_log_ratio", fit.anisotropy_log_ratio)),
        "max_neighbor_overlap": _finite("max_neighbor_overlap", fit.max_neighbor_overlap),
        "geocode_rooftop": _binary("geocode_rooftop", geocode_rooftop),
        "height_source_authoritative": _binary("height_source_authoritative", height_source_authoritative),
    }


def _select(values, names):
    unknown = [n for n in names if n not in values]
    if unknown:
        raise KeyError(f"unknown feature(s) {unknown}; known: {list(FEATURE_NAMES)}")
    return np.array([values[n] for n in names], dtype=float)


def feature_vector(fit, photo, fp, *, geocode_rooftop, height_source_authoritative, names=FEATURE_NAMES):
    """Features as a 1-D float array in `names` order."""
    return _select(
        feature_dict(fit, photo, fp, geocode_rooftop=geocode_rooftop, height_source_authoritative=height_source_authoritative),
        names,
    )


def feature_matrix(rows, names=FEATURE_NAMES):
    """(n_rows, len(names)) matrix from labelled rows exposing .fit .photo .footprint
    .geocode_rooftop .height_source_authoritative (e.g. fixtures.fake_fits.LabelledFit)."""
    return np.vstack(
        [
            feature_vector(
                r.fit, r.photo, r.footprint,
                geocode_rooftop=r.geocode_rooftop, height_source_authoritative=r.height_source_authoritative, names=names,
            )
            for r in rows
        ]
    )
