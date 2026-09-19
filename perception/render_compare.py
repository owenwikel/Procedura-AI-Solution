#!/usr/bin/env python
"""Render-and-compare orientation scoring (ML_ADDENDUM A.3 stages 2-3, SPEC 6.1(b) / 6.6 Filter 4).

    result = render_compare(mesh, mask)            # trimesh mesh + mask PNG path or bool array
    result.best, result.second, result.margin, result.candidates

The mesh is rendered as an orthographic *silhouette* from every candidate
(up-axis x 4 azimuths) and each silhouette is scored against the photo's building
mask by normalised-shape IoU: both shapes are cropped to their bounding box,
uniformly scaled so the longer side fills a common square canvas, and centred
before the IoU. That discards position and scale (which a perspective photo
corrupts) and keeps aspect ratio and profile (which discriminate orientation).

Conventions
  * Mesh frame is glTF: +Y up, front facing +Z. Candidate up-axes default to the six
    signed coordinate axes; pass the top 2-3 from SPEC 6.1(a) to prune.
  * For up-axis u, azimuth phi places the camera at R_u(phi) . e1 looking at the mesh
    (right-handed rotation about u), where e1 is the coordinate axis after u's dominant
    axis in X->Y->Z->X order, made perpendicular to u. For u=+Y, phi=0 the camera sits on
    +Z (sees the glTF front facade) with +X to the image right; increasing phi swings the
    camera counter-clockwise seen from above. Candidate k is phi = theta0 + k*90 deg.
  * A candidate scores how well the mesh, viewed from that side with that up-axis,
    matches the photo. Mapping it to a footprint rotation is the placement solver's job.

margin = score(best) - score(second best), as in SPEC 6.6 (< 0.05 -> manual review).
Silhouettes are mirror-symmetric under a 180 deg change of view, so a left-right
symmetric mesh ties with its own back view and a box has margin 0; `ties_with_best`
counts those exact ties.

Usage: python perception/render_compare.py MESH.glb MASK.png [--top N]
"""
import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

CANVAS_PX = 128
PAD_PX = 2
SUBPIXEL_BITS = 4  # cv2.fillConvexPoly fixed-point precision
TIE_TOL = 1e-9

# Y-up (glTF) first so exact ties resolve to the glTF-canonical orientation.
SIGNED_AXES = [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)]


@dataclass(frozen=True)
class Candidate:
    up_axis: tuple
    azimuth_deg: float
    score: float


@dataclass(frozen=True)
class Comparison:
    candidates: list  # every Candidate, best first
    best: Candidate
    second: Candidate
    margin: float
    ties_with_best: int  # other candidates scoring identically to best (symmetry, not evidence)

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------------ inputs


def _as_mesh(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())  # dump() bakes scene-graph transforms in
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected a trimesh.Trimesh or Scene, got {type(mesh).__name__}")
    if len(mesh.faces) == 0:
        raise ValueError("mesh has no faces")
    return mesh


def _as_mask(mask):
    if isinstance(mask, (str, Path)):
        mask = np.array(Image.open(mask).convert("L")) > 127
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    if not mask.any():
        raise ValueError("mask is empty")
    return mask


# ---------------------------------------------------------- normalisation


def normalise_mask(mask, size=CANVAS_PX):
    """Crop to the bounding box, scale uniformly so the longer side fits, centre on a size x size canvas."""
    ys, xs = np.nonzero(mask)
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.float32)
    h, w = crop.shape
    scale = (size - 2 * PAD_PX) / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA) >= 0.5
    canvas = np.zeros((size, size), bool)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    canvas[oy : oy + nh, ox : ox + nw] = small
    return canvas


def camera_basis(up_axis, azimuth_rad):
    """Returns (right, up, toward_camera) unit vectors for the convention in the module docstring."""
    up = np.asarray(up_axis, dtype=float)
    norm = np.linalg.norm(up)
    if norm == 0:
        raise ValueError("up axis must be non-zero")
    up = up / norm
    ref = np.eye(3)[(int(np.argmax(np.abs(up))) + 1) % 3]
    e1 = ref - ref.dot(up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    toward = np.cos(azimuth_rad) * e1 + np.sin(azimuth_rad) * e2  # camera sits here, looking back at the mesh
    right = np.cross(-toward, up)
    return right, up, toward


def render_silhouette(mesh, up_axis, azimuth_rad, size=CANVAS_PX):
    """Orthographic silhouette of `mesh`, already normalised to the common canvas (bool array)."""
    right, up, _ = camera_basis(up_axis, azimuth_rad)
    xy = mesh.vertices @ np.stack([right, up], axis=1)  # (V, 2): image-x, image-up
    tris = xy[mesh.faces]  # (F, 3, 2)
    lo, hi = tris.reshape(-1, 2).min(axis=0), tris.reshape(-1, 2).max(axis=0)
    extent = float((hi - lo).max())
    canvas = np.zeros((size, size), np.uint8)
    if extent == 0:
        return canvas.astype(bool)
    scale = (size - 2 * PAD_PX) / extent
    off = (size - (hi - lo) * scale) / 2  # centre the shorter side
    px = (tris[..., 0] - lo[0]) * scale + off[0]
    py = (hi[1] - tris[..., 1]) * scale + off[1]  # image y grows downward
    pts = np.stack([px, py], axis=-1) - 0.5  # edge convention -> pixel-centre convention
    # Triangles are filled one at a time: cv2.fillPoly fills all its polygons in one even-odd
    # pass, so overlapping triangles (front and back faces of any closed mesh) would cancel out.
    # Sub-pixel triangles dominate dense meshes, so mark the pixels under their vertices in one
    # vectorised step and only rasterise the rest.
    tiny = (pts.max(axis=1) - pts.min(axis=1)).max(axis=1) <= 1.0
    corners = np.clip(np.round(pts[tiny].reshape(-1, 2)).astype(int), 0, size - 1)
    canvas[corners[:, 1], corners[:, 0]] = 1
    for tri in np.round(pts[~tiny] * (1 << SUBPIXEL_BITS)).astype(np.int32):
        cv2.fillConvexPoly(canvas, tri, 1, shift=SUBPIXEL_BITS)
    return canvas.astype(bool)


def normalised_iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


# ------------------------------------------------------------------- scoring


def render_compare(mesh, mask, up_axes=None, theta0=0.0, size=CANVAS_PX):
    """Score every (up-axis, azimuth) candidate against the photo mask. See the module docstring."""
    mesh = _as_mesh(mesh)
    target = normalise_mask(_as_mask(mask), size)
    axes = SIGNED_AXES if up_axes is None else [tuple(float(v) for v in a) for a in up_axes]
    if not axes:
        raise ValueError("up_axes is empty")
    scored = []
    for axis in axes:
        for k in range(4):
            phi = theta0 + k * np.pi / 2
            sil = render_silhouette(mesh, axis, phi, size)
            scored.append(Candidate(tuple(axis), float(np.degrees(phi) % 360), normalised_iou(sil, target)))
    ranked = sorted(scored, key=lambda c: -c.score)  # stable: ties keep enumeration order
    best, second = ranked[0], ranked[1]
    ties = sum(abs(c.score - best.score) <= TIE_TOL for c in ranked[1:])
    return Comparison(ranked, best, second, best.score - second.score, ties)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mesh")
    parser.add_argument("mask")
    parser.add_argument("--top", type=int, default=8, help="how many candidates to print")
    args = parser.parse_args()
    result = render_compare(trimesh.load(args.mesh, force="mesh"), args.mask)
    for c in result.candidates[: args.top]:
        print(f"up={c.up_axis!s:<14} azimuth={c.azimuth_deg:5.1f}  score={c.score:.3f}")
    print(f"margin={result.margin:.3f}  ties_with_best={result.ties_with_best}")
    if result.margin < 0.05:
        print("margin < 0.05: route to review (SPEC 6.6)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
