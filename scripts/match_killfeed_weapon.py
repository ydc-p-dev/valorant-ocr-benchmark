#!/usr/bin/env python3
"""
Find a weapon silhouette template inside the killfeed region of a screenshot.

Uses masked normalized cross-correlation (TM_CCORR_NORMED) so transparent
template pixels (from process_weapon_icon.py output) do not bias the score.

With multiple killfeed rows, **white-weapon slot** mode (default) thresholds bright
pixels, finds a gun-shaped connected component between left/right name regions, and
runs template matching **only inside that X window** — avoiding agent portraits and
much of the text.

**False positives:** a killfeed weapon silhouette is short on fine detail; many
background patches share similar brightness under the mask, so NCC can still
report 0.94–0.98 there. With a **narrow white-weapon slot**, rifle templates may
not fit the crop; class scores for other guns must be filled using the **full
row center strip** for supplement NCC, and the **label must come from peaks
inside the match window** (supplement is only for the runner-up margin).
With multiple row strips (``--row-bands-json`` or
``--within-kill-rows``), the script defaults to **one classification per row**
(``--max-per-row``) using a horizontal **weapon-column anchor** (see
``--weapon-anchor-x-frac``): Valorant’s gun icon sits **right of** the agent
portrait, so anchoring at strip midpoint (0.5) often latches onto the portrait.
Use ``--classify-vote-min-x-frac`` so weapon **labels** ignore peaks left of the portrait.
Also try a tighter ``--center-frac``, a stricter ``--min-score``, or fewer
``--scales`` if the true icon only matches at one scale.

**Many weapons:** use ``--templates-json`` (map id → template path). For each row the
script takes the best NCC peak near the weapon column, then compares **max score per
weapon** in a small radius. The label wins only if ``best - second_best`` ≥
``--min-class-margin`` (Phantom vs Vandal silhouettes often differ by ~0.02–0.08).
Use ``--only-weapon Vandal`` to drop rows where another gun wins.

Examples:

  python scripts/match_killfeed_weapon.py killfeed_screenshots/foo.png \\
      --template assets/processed/Vandal_icon_template.png

  python scripts/match_killfeed_weapon.py screen.png -t assets/processed/Vandal_template.png \\
      --killfeed-rect 80,1300,600,180 --draw out_debug.png

  python scripts/match_killfeed_weapon.py strip.png -t Vandal.png --no-crop

  python scripts/match_killfeed_weapon.py killfeed_screenshots/killfeed_20260319_193608_606.png \\
      -t assets/processed/Vandal_icon_template.png --within-kill-rows --draw /tmp/hits.png

  python scripts/match_killfeed_weapon.py screen.png -t Vandal_template.png \\
      --row-bands-json config/killfeed_row_bands.json --center-frac 0.35 --draw /tmp/hits.png

  python scripts/match_killfeed_weapon.py screen.png \\
      --templates-json config/weapon_templates.example.json \\
      --row-bands-json config/killfeed_row_bands.json --center-frac 0.35 --draw /tmp/labeled.png

  python scripts/match_killfeed_weapon.py screen.png -t Vandal.png \\
      --row-bands-json config/killfeed_row_bands.json --bench-repetitions 10 --quiet
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from valorant_killfeed_tracker import (
    REGION_KILLFEED,
    crop_killfeed_region_if_possible,
    detect_killfeed_row_boxes,
    load_row_bands_json,
    split_tall_killfeed_row_boxes,
)


def roi_from_image(
    full: np.ndarray,
    *,
    killfeed_rect: dict[str, int] | None,
    no_crop: bool,
) -> tuple[np.ndarray, int, int]:
    """Return (roi_bgr, offset_x, offset_y) in full-image coordinates."""
    if no_crop:
        return full, 0, 0
    rect = killfeed_rect if killfeed_rect is not None else REGION_KILLFEED
    h, w = full.shape[:2]
    x, y = rect["left"], rect["top"]
    rw, rh = rect["width"], rect["height"]
    if w >= x + rw and h >= y + rh:
        return full[y : y + rh, x : x + rw], x, y
    return apply_killfeed_rect(full, rect), 0, 0


def parse_killfeed_rect(s: str) -> dict[str, int]:
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 4 or any(p < 0 for p in parts):
        raise argparse.ArgumentTypeError("expected TOP,LEFT,WIDTH,HEIGHT (four non-negative ints)")
    return {"top": parts[0], "left": parts[1], "width": parts[2], "height": parts[3]}


def apply_killfeed_rect(img: np.ndarray, rect: dict[str, int]) -> np.ndarray:
    h, w = img.shape[:2]
    x = max(0, min(rect["left"], w - 1))
    y = max(0, min(rect["top"], h - 1))
    rw = max(1, min(rect["width"], w - x))
    rh = max(1, min(rect["height"], h - y))
    return img[y : y + rh, x : x + rw]


def patches_from_row_bands_frac(
    gray: np.ndarray,
    bands: list[tuple[float, float]],
    center_frac: float,
    roi_w: int,
    roi_h: int,
) -> tuple[list[tuple[np.ndarray, int, int]], list[tuple[int, int, int, int]]]:
    """Build search patches and orange debug boxes (full-width strips in ROI coords)."""
    patches: list[tuple[np.ndarray, int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []
    for y0f, y1f in bands:
        y0 = min(roi_h - 2, max(0, int(roi_h * y0f)))
        y1 = min(roi_h, max(y0 + 2, int(round(roi_h * y1f))))
        sub = gray[y0:y1, :]
        sg, ix = center_strip(sub, center_frac)
        patches.append((sg, ix, y0))
        boxes.append((0, y0, roi_w, y1 - y0))
    return patches, boxes


def filter_plausible_row_boxes(
    rows: list[tuple[int, int, int, int, str]],
    roi_w: int,
    roi_h: int,
    *,
    min_width_frac: float,
    max_center_y_frac: float | None,
) -> list[tuple[int, int, int, int, str]]:
    """
    Drop HSV false positives: green mask often matches terrain in the lower part of the
    killfeed ROI (narrow blobs), while real rows span most of the strip and sit higher.
    """
    if not rows:
        return []
    mwf = max(0.15, min(0.95, min_width_frac))
    min_w = max(120, int(roi_w * mwf))
    out: list[tuple[int, int, int, int, str]] = []
    for x, y, w, h, c in rows:
        if w < min_w:
            continue
        if max_center_y_frac is not None:
            cy = y + h * 0.5
            limit = roi_h * max(0.2, min(0.95, max_center_y_frac))
            if cy >= limit:
                continue
        out.append((x, y, w, h, c))
    return out


def clamp_row_box(
    x: int, y: int, w: int, h: int, pad: int, roi_w: int, roi_h: int
) -> tuple[int, int, int, int]:
    """Expand box by ``pad`` and clip to ROI; returns (x, y, w, h) with w,h >= 1."""
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(roi_w, x + w + pad)
    y1 = min(roi_h, y + h + pad)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def center_strip(gray: np.ndarray, center_frac: float) -> tuple[np.ndarray, int]:
    """If center_frac > 0, return (cropped_gray, x0 in parent); else (gray, 0)."""
    if center_frac <= 0:
        return gray, 0
    cf = min(0.49, max(0.05, center_frac))
    _, W = gray.shape[:2]
    x0 = int(W * (0.5 - cf))
    x1 = int(W * (0.5 + cf))
    x0 = max(0, x0)
    x1 = min(W, max(x0 + 2, x1))
    return gray[:, x0:x1], x0


def white_weapon_slot_x_slice(
    gray: np.ndarray,
    *,
    white_thr: int,
    cx_min_frac: float,
    cx_max_frac: float,
    min_area: int,
    max_area: int,
    min_aspect: float,
    max_aspect: float,
    min_h_frac: float,
    max_h_frac: float,
    pad_px: int,
) -> tuple[int, int] | None:
    """
    Find horizontal [x0, x1) slice where the killfeed **white gun icon** likely sits:
    one connected component of bright pixels, wider than tall, between name columns.

    Coordinates are relative to ``gray`` (one row strip, already center-cropped if any).
    """
    H, W = gray.shape[:2]
    if W < 32 or H < 8:
        return None
    thr = max(1, min(255, int(white_thr)))
    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    lo = max(0.05, min(0.45, float(cx_min_frac))) * W
    hi = max(0.55, min(0.95, float(cx_max_frac))) * W
    best_i = -1
    best_score = -1.0
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        if bw < 8 or bh < 5:
            continue
        h_min = max(5, int(H * min_h_frac))
        h_max = min(H, int(round(H * max_h_frac))) + 2
        if bh < h_min or bh > h_max:
            continue
        asp = bw / float(max(bh, 1))
        if asp < min_aspect or asp > max_aspect:
            continue
        cx = x + bw * 0.5
        if cx < lo or cx > hi:
            continue
        mid = 0.52 * W
        mid_bonus = 1.0 + 0.35 * math.exp(-((cx - mid) ** 2) / (0.18 * W) ** 2)
        shape_bonus = 1.0 - min(abs(asp - 2.35), 2.5) * 0.07
        score = float(area) * mid_bonus * max(0.35, shape_bonus)
        if score > best_score:
            best_score = score
            best_i = i
    if best_i < 0:
        return None
    x = int(stats[best_i, cv2.CC_STAT_LEFT])
    bw = int(stats[best_i, cv2.CC_STAT_WIDTH])
    pad = max(0, int(pad_px))
    x0 = max(0, x - pad)
    x1 = min(W, x + bw + pad)
    if x1 <= x0 + 4:
        return None
    return (x0, x1)


def min_slot_dims_for_all_templates(
    templates: dict[str, tuple[np.ndarray, np.ndarray]],
    scales: list[float],
) -> tuple[int, int]:
    """
    Smallest crop size so **each** weapon template fits at **at least one** scale.

    Using max over scales would force huge crops (e.g. 0.45× wide rifle), pulling in
    portrait/background and shrinking Phantom–Vandal margin below ``--min-class-margin``.
    """
    need_w, need_h = 2, 2
    for templ, _mask in templates.values():
        th0, tw0 = templ.shape[:2]
        w_min = 10**9
        h_min = 10**9
        for sc in scales:
            if sc <= 0:
                continue
            tw = max(2, int(round(tw0 * sc)))
            th = max(2, int(round(th0 * sc)))
            w_min = min(w_min, tw)
            h_min = min(h_min, th)
        need_w = max(need_w, w_min)
        need_h = max(need_h, h_min)
    return need_w, need_h


def expand_x_slice_to_min_width(sx0: int, sx1: int, strip_w: int, min_w: int) -> tuple[int, int]:
    """Widen [sx0, sx1) to at least ``min_w`` pixels, centered when possible, clipped to [0, strip_w)."""
    w = sx1 - sx0
    if w >= min_w or strip_w < min_w:
        return sx0, sx1
    need = min_w - w
    cx = (sx0 + sx1) // 2
    n0 = max(0, cx - min_w // 2)
    n1 = min(strip_w, n0 + min_w)
    if n1 - n0 < min_w:
        n0 = max(0, strip_w - min_w)
        n1 = strip_w
    return n0, n1


def load_templates_json(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    JSON object: weapon id (string) → path to silhouette PNG (relative to the JSON file
    unless absolute). Example: ``{\"Vandal\": \"../assets/processed/Vandal_icon_template.png\"}``.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise SystemExit(f"Cannot read templates JSON: {path} ({e})") from e
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON: {path}: {e}") from e
    if not isinstance(raw, dict) or not raw:
        raise SystemExit(f"Templates JSON must be a non-empty object of id: path strings: {path}")
    base = path.parent
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not k.strip():
            raise SystemExit(f"Invalid weapon id in {path}: {k!r}")
        if not isinstance(v, str) or not v.strip():
            raise SystemExit(f"Invalid template path for {k!r} in {path}")
        p = Path(v.strip())
        if not p.is_absolute():
            p = (base / p).resolve()
        if not p.is_file():
            raise SystemExit(f"Template file not found for {k!r}: {p}")
        out[k.strip()] = load_template(p)
    return out


def load_template(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (template_gray, mask_uint8) for matchTemplate(..., mask=mask)."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise SystemExit(f"Cannot read template: {path}")
    if im.ndim == 2:
        templ = im
        mask = (templ > 8).astype(np.uint8) * 255
    elif im.shape[2] == 4:
        a = im[:, :, 3]
        templ = a.astype(np.uint8)
        mask = (a > 8).astype(np.uint8) * 255
    else:
        templ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        mask = np.full_like(templ, 255, dtype=np.uint8)
    if cv2.countNonZero(mask) < 10:
        raise SystemExit("Template mask has almost no pixels — check alpha / image content.")
    return templ, mask


HitTagged = tuple[str, int, int, int, int, float, float]


def merge_tagged_hits_across_scales(
    hits: list[HitTagged],
    *,
    merge_dist_frac: float,
    max_matches: int,
) -> list[HitTagged]:
    if not hits:
        return []
    shifted = sorted(hits, key=lambda t: t[5], reverse=True)
    merged: list[HitTagged] = []
    mdf = max(0.05, min(0.95, merge_dist_frac))
    for w, x, y, tw, th, score, sc in shifted:
        cx, cy = x + tw // 2, y + th // 2
        ok = True
        for _w2, x2, y2, tw2, th2, _, _ in merged:
            c2x, c2y = x2 + tw2 // 2, y2 + th2 // 2
            need = mdf * max(tw, th, tw2, th2)
            if math.hypot(cx - c2x, cy - c2y) < need:
                ok = False
                break
        if ok:
            merged.append((w, x, y, tw, th, score, sc))
        if len(merged) >= max_matches:
            break
    return merged


def max_scores_by_weapon_near(
    patch_hits: list[HitTagged],
    cx: int,
    cy: int,
    radius_px: float,
) -> dict[str, float]:
    """Best NCC seen per weapon id among peaks whose box center lies within radius of (cx, cy)."""
    by_w: dict[str, float] = {}
    r2 = radius_px * radius_px
    for w, x, y, tw, th, sc, _ in patch_hits:
        ccx = x + tw // 2
        ccy = y + th // 2
        dx, dy = ccx - cx, ccy - cy
        if dx * dx + dy * dy > r2:
            continue
        prev = by_w.get(w)
        if prev is None or sc > prev:
            by_w[w] = sc
    return by_w


def max_scores_by_weapon_right_of_strip(
    patch_hits: list[HitTagged],
    *,
    base_x: int,
    strip_w: int,
    min_center_x_frac: float,
) -> dict[str, float]:
    """
    Best NCC per weapon among peaks whose box center is at or right of
    ``base_x + strip_w * min_center_x_frac`` (excludes agent portrait on the left).
    """
    mxf = max(0.15, min(0.85, min_center_x_frac))
    x_min = base_x + int(round(strip_w * mxf))
    by_w: dict[str, float] = {}
    for w, x, y, tw, th, sc, _ in patch_hits:
        if x + tw // 2 < x_min:
            continue
        prev = by_w.get(w)
        if prev is None or sc > prev:
            by_w[w] = sc
    return by_w


def best_geometry_for_weapon(
    patch_hits: list[HitTagged],
    weapon: str,
    cx: int,
    cy: int,
    radius_px: float,
) -> HitTagged | None:
    """Highest-scoring peak for ``weapon`` near (cx, cy) (for stable box size/position)."""
    r2 = radius_px * radius_px
    best: HitTagged | None = None
    for h in patch_hits:
        w, x, y, tw, th, sc, sc2 = h
        if w != weapon:
            continue
        ccx = x + tw // 2
        ccy = y + th // 2
        dx, dy = ccx - cx, ccy - cy
        if dx * dx + dy * dy > r2:
            continue
        if best is None or sc > best[5]:
            best = h
    return best


def nms_peaks(
    response: np.ndarray,
    templ_w: int,
    templ_h: int,
    *,
    min_score: float,
    max_matches: int,
    min_dist_frac: float,
) -> list[tuple[int, int, float]]:
    """Greedy suppression on matchTemplate heatmap (TM_CCORR_NORMED, higher=better)."""
    res = response.copy()
    # Wide silhouettes need margin from max side; using only min(w,h) leaves dense peaks.
    span = max(templ_w, templ_h)
    min_dist = max(6, int(span * min_dist_frac))
    out: list[tuple[int, int, float]] = []
    for _ in range(max_matches):
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < min_score:
            break
        x, y = max_loc
        out.append((x, y, float(max_val)))
        x0 = max(0, x - min_dist)
        y0 = max(0, y - min_dist)
        x1 = min(res.shape[1], x + templ_w + min_dist)
        y1 = min(res.shape[0], y + templ_h + min_dist)
        res[y0:y1, x0:x1] = 0.0
    return out


def match_at_scale(
    gray: np.ndarray,
    templ: np.ndarray,
    mask: np.ndarray,
    scale: float,
    *,
    min_score: float,
    max_matches: int,
    min_dist_frac: float,
) -> list[tuple[int, int, int, int, float, float]]:
    """
    Each hit: (x, y, tw, th, score, scale).
    Coordinates are relative to ``gray`` (killfeed ROI).
    """
    if scale <= 0:
        return []
    th0, tw0 = templ.shape[:2]
    tw = max(2, int(round(tw0 * scale)))
    th = max(2, int(round(th0 * scale)))
    if tw > gray.shape[1] or th > gray.shape[0]:
        return []
    t = cv2.resize(templ, (tw, th), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
    m = (m > 127).astype(np.uint8) * 255
    if cv2.countNonZero(m) < 5:
        return []
    res = cv2.matchTemplate(gray, t, cv2.TM_CCORR_NORMED, mask=m)
    peaks = nms_peaks(res, tw, th, min_score=min_score, max_matches=max_matches, min_dist_frac=min_dist_frac)
    return [(x, y, tw, th, sc, scale) for x, y, sc in peaks]


def max_template_ncc_global(
    gray: np.ndarray,
    templ: np.ndarray,
    mask: np.ndarray,
    scale: float,
) -> float | None:
    """Global max TM_CCORR_NORMED over ``gray`` for one scaled template (any position)."""
    if scale <= 0:
        return None
    th0, tw0 = templ.shape[:2]
    tw = max(2, int(round(tw0 * scale)))
    th = max(2, int(round(th0 * scale)))
    if tw > gray.shape[1] or th > gray.shape[0]:
        return None
    t = cv2.resize(templ, (tw, th), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
    m = (m > 127).astype(np.uint8) * 255
    if cv2.countNonZero(m) < 5:
        return None
    res = cv2.matchTemplate(gray, t, cv2.TM_CCORR_NORMED, mask=m)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return float(max_val)


def supplement_class_scores_from_gray(
    by_w: dict[str, float],
    match_gray: np.ndarray,
    templates: dict[str, tuple[np.ndarray, np.ndarray]],
    scales: list[float],
) -> None:
    """
    Add scores for weapons missing from ``by_w`` using global max NCC on ``match_gray``.

    Pass the **full row center strip** when the match window is a narrow white slot: rifle templates
    may not fit the slot width, which would otherwise leave other classes at 0.0 and fake a huge
    class margin (false Spectre, etc.).
    """
    for wname, (templ, mask) in templates.items():
        if wname in by_w:
            continue
        best = 0.0
        for sc in scales:
            v = max_template_ncc_global(match_gray, templ, mask, sc)
            if v is not None and v > best:
                best = v
        by_w[wname] = best


@dataclass
class WeaponHit:
    """One weapon template match aligned to ``row_bands_frac`` index."""

    band_index: int
    weapon: str
    x: int
    y: int
    w: int
    h: int
    score: float
    scale: float
    margin: float
    vs_weapon: str


@dataclass
class WeaponMatchParams:
    """Defaults tuned with ``config/killfeed_row_bands.json`` + multi-template JSON."""

    center_frac: float = 0.35
    scales: str = "0.25,0.35,0.45"
    min_score: float = 0.25
    min_class_margin: float = 0.011
    min_weapons_for_classify: int = 1
    supplement_class_scores: bool = True
    only_weapon: str | None = None
    classify_radius_px: float = 0.0
    max_matches: int = 8
    peak_nms_frac: float = 0.28
    merge_dist_frac: float = 0.42
    weapon_anchor_x_frac: float = 0.62
    classify_vote_min_x_frac: float = 0.42
    weapon_white_slot: bool = True
    weapon_white_thr: int = 205
    weapon_slot_pad: int = 14
    weapon_slot_cx_min_frac: float = 0.14
    weapon_slot_cx_max_frac: float = 0.86


def match_weapons_in_roi_bands(
    roi_bgr: np.ndarray,
    bands: list[tuple[float, float]],
    templates_json: Path,
    params: WeaponMatchParams | None = None,
    band_active: list[bool] | None = None,
) -> list[WeaponHit | None]:
    """
    Run weapon template matching once per row band (same geometry as OCR with fixed bands).
    Returns a list parallel to ``bands`` (``None`` if that row did not pass classification gates).

    If ``band_active`` is set (same length as bands / patches), inactive bands skip template work
    (e.g. empty killfeed slots over map background).
    """
    p = params or WeaponMatchParams()
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    roi_h, roi_w = gray.shape[:2]
    patches, _ = patches_from_row_bands_frac(gray, bands, p.center_frac, roi_w, roi_h)
    templates = load_templates_json(templates_json.resolve())
    scales = [float(s.strip()) for s in p.scales.split(",") if s.strip()]
    multitpl = len(templates) > 1
    classify_r = float(p.classify_radius_px)
    if classify_r <= 0:
        classify_r = max(28.0, 0.06 * float(roi_w))
    only_w = p.only_weapon.strip() if p.only_weapon else None
    weapon_anchor_frac = max(0.35, min(0.85, float(p.weapon_anchor_x_frac)))
    vote_min_x_frac = float(p.classify_vote_min_x_frac)

    def classify_row_patch(
        patch_hits: list[HitTagged],
        *,
        base_x: int,
        strip_w: int,
        anchor_x: int,
        anchor_y: int,
        vote_x_frac: float,
        ncc_radius: float | None = None,
        match_gray: np.ndarray | None = None,
        supplement_gray: np.ndarray | None = None,
    ) -> list[tuple[str, int, int, int, int, float, float, float, str]]:
        r = classify_r if ncc_radius is None else ncc_radius
        out: list[tuple[str, int, int, int, int, float, float, float, str]] = []
        vxf = max(0.0, min(0.85, float(vote_x_frac)))
        by_w = max_scores_by_weapon_right_of_strip(
            patch_hits,
            base_x=base_x,
            strip_w=strip_w,
            min_center_x_frac=vxf,
        )
        if not by_w:
            by_w = max_scores_by_weapon_near(patch_hits, anchor_x, anchor_y, r)
        if not by_w:
            return out
        peaks_by_w = dict(by_w)
        if multitpl and match_gray is not None and p.supplement_class_scores:
            gray_sup = supplement_gray if supplement_gray is not None else match_gray
            supplement_class_scores_from_gray(by_w, gray_sup, templates, scales)
        if multitpl and len(by_w) < max(1, int(p.min_weapons_for_classify)):
            return out
        if multitpl:
            win_w, win_s = max(peaks_by_w.items(), key=lambda kv: kv[1])
            others = [wid for wid in templates if wid != win_w]
            if others:
                sec_w, sec_s = max(((w, by_w.get(w, 0.0)) for w in others), key=lambda kv: kv[1])
            else:
                sec_w, sec_s = "", 0.0
            margin = (win_s - sec_s) if sec_w else 0.0
        else:
            ranked = sorted(by_w.items(), key=lambda kv: kv[1], reverse=True)
            win_w, win_s = ranked[0]
            sec_w, sec_s = ranked[1] if len(ranked) > 1 else ("", 0.0)
            margin = (win_s - sec_s) if sec_w else 0.0

        if win_s < p.min_score:
            return out
        if multitpl and p.min_class_margin > 0:
            need_m = float(p.min_class_margin)
            if margin < need_m:
                if not (win_s >= 0.982 and margin >= need_m - 0.00085):
                    return out
        if only_w is not None and win_w != only_w:
            return out

        geom = best_geometry_for_weapon(patch_hits, win_w, anchor_x, anchor_y, r)
        if geom is None:
            x_cut = base_x + int(round(strip_w * max(0.15, min(0.85, vxf))))
            right_only = [
                h for h in patch_hits if h[0] == win_w and h[1] + h[3] // 2 >= x_cut
            ]
            geom = max(right_only, key=lambda h: h[5]) if right_only else None
        if geom is None:
            same_id = [h for h in patch_hits if h[0] == win_w]
            geom = max(same_id, key=lambda h: h[5]) if same_id else None
        if geom is None:
            return out
        w_g, x, y, tw, th, sc_geom, sc_fn = geom
        out.append((w_g, x, y, tw, th, sc_geom, sc_fn, margin, sec_w))
        return out

    hits_out: list[WeaponHit | None] = [None] * len(patches)
    if not patches:
        return hits_out

    inner_cap = max(p.max_matches, 8, 32)
    use_white_slot = bool(p.weapon_white_slot) and len(patches) > 1

    for band_i, (search_gray, base_x, base_y) in enumerate(patches):
        if band_active is not None and band_i < len(band_active) and not band_active[band_i]:
            continue
        strip_w, strip_h = search_gray.shape[1], search_gray.shape[0]
        slot_slice: tuple[int, int] | None = None
        if use_white_slot:
            Hs, Ws = strip_h, strip_w
            min_a = max(35, int(Hs * Ws * 0.0015))
            max_a = min(12000, int(Hs * Ws * 0.12))
            slot_slice = white_weapon_slot_x_slice(
                search_gray,
                white_thr=p.weapon_white_thr,
                cx_min_frac=p.weapon_slot_cx_min_frac,
                cx_max_frac=p.weapon_slot_cx_max_frac,
                min_area=min_a,
                max_area=max_a,
                min_aspect=1.12,
                max_aspect=7.5,
                min_h_frac=0.26,
                max_h_frac=0.99,
                pad_px=max(0, p.weapon_slot_pad),
            )
        if slot_slice is not None:
            sx0, sx1 = slot_slice
            min_tw, _min_th = min_slot_dims_for_all_templates(templates, scales)
            if strip_w >= min_tw:
                sx0, sx1 = expand_x_slice_to_min_width(sx0, sx1, strip_w, min_tw)
            match_gray = search_gray[:, sx0:sx1]
            eff_base_x = base_x + sx0
            eff_w = sx1 - sx0
            anchor_x = eff_base_x + eff_w // 2
            vote_xf = 0.06
            half_diag = math.hypot(float(eff_w) * 0.5, float(strip_h) * 0.5)
            local_r = max(
                14.0,
                half_diag * 0.98,
                min(float(classify_r), 0.25 * float(strip_w)),
            )
        else:
            match_gray = search_gray
            eff_base_x = base_x
            eff_w = strip_w
            anchor_x = base_x + int(round(strip_w * weapon_anchor_frac))
            vote_xf = vote_min_x_frac
            local_r = None

        anchor_y = base_y + strip_h // 2
        patch_hits: list[HitTagged] = []
        for wname, (templ, mask) in templates.items():
            for sc in scales:
                for x, y, tw, th, score, sc2 in match_at_scale(
                    match_gray,
                    templ,
                    mask,
                    sc,
                    min_score=p.min_score,
                    max_matches=inner_cap,
                    min_dist_frac=p.peak_nms_frac,
                ):
                    patch_hits.append((wname, eff_base_x + x, base_y + y, tw, th, score, sc2))

        row_hits = classify_row_patch(
            patch_hits,
            base_x=eff_base_x,
            strip_w=eff_w,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            vote_x_frac=vote_xf,
            ncc_radius=local_r,
            match_gray=match_gray,
            supplement_gray=search_gray,
        )
        if row_hits:
            t0 = row_hits[0]
            hits_out[band_i] = WeaponHit(
                band_index=band_i,
                weapon=t0[0],
                x=t0[1],
                y=t0[2],
                w=t0[3],
                h=t0[4],
                score=t0[5],
                scale=t0[6],
                margin=t0[7],
                vs_weapon=t0[8],
            )

    return hits_out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Template-match a weapon icon inside the killfeed region of an image."
    )
    p.add_argument("image", type=Path, help="Screenshot (full monitor) or killfeed crop.")
    p.add_argument(
        "-t",
        "--template",
        type=Path,
        default=None,
        help="Single silhouette PNG (BGRA or mask). Weapon id in output is the file stem. "
        "Omit if using --templates-json.",
    )
    p.add_argument(
        "--templates-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="JSON object: weapon id → relative path to template PNG (paths relative to this file). "
        "When 2+ weapons are listed, --min-class-margin filters ambiguous rows.",
    )
    p.add_argument(
        "--killfeed-rect",
        type=parse_killfeed_rect,
        default=None,
        metavar="TOP,LEFT,WIDTH,HEIGHT",
        help=f"Override killfeed rect (full-image coords). Default: {REGION_KILLFEED['top']},"
        f"{REGION_KILLFEED['left']},{REGION_KILLFEED['width']},{REGION_KILLFEED['height']}",
    )
    p.add_argument(
        "--no-crop",
        action="store_true",
        help="Use the whole image as the search ROI (already a killfeed strip).",
    )
    p.add_argument(
        "--center-frac",
        type=float,
        default=0.0,
        metavar="0..1",
        help="If >0, keep only the central horizontal band (weapon column): on full ROI, or inside each row with --within-kill-rows.",
    )
    p.add_argument(
        "--row-bands-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="JSON with 'row_bands_frac': [[y0,y1], ...] as fractions of killfeed ROI height (0=top). "
        "Overrides --within-kill-rows. Copy from config/killfeed_row_bands.example.json and tune.",
    )
    p.add_argument(
        "--within-kill-rows",
        action="store_true",
        help="Search inside green/red HSV row bands (same as tracker). Narrow/low blobs are filtered; "
        "if none remain, falls back to full ROI with a warning.",
    )
    p.add_argument(
        "--row-pad",
        type=int,
        default=6,
        help="Padding in px around each detected row box (only with --within-kill-rows).",
    )
    p.add_argument(
        "--min-row-width-frac",
        type=float,
        default=0.40,
        metavar="0..1",
        help="With --within-kill-rows: min row width as fraction of ROI width (excludes terrain splotches).",
    )
    p.add_argument(
        "--max-row-center-y-frac",
        type=float,
        default=0.78,
        metavar="0..1",
        help="With --within-kill-rows: keep rows whose vertical center is above this fraction of ROI height "
        "(excludes ground at the bottom of the strip). Use 0 to disable.",
    )
    p.add_argument("--min-score", type=float, default=0.38, help="Min TM_CCORR_NORMED peak (0–1).")
    p.add_argument(
        "--min-class-margin",
        type=float,
        default=0.0,
        metavar="Δ",
        help="With 2+ templates: require (best NCC − second best) ≥ this or skip the row (0 = off). "
        "After --weapon-white-slot, scores are often only ~0.002–0.008 apart; raise templates quality "
        "then try e.g. 0.006–0.015. Very strong winners (NCC ≥ 0.982) get a 0.00085 slack vs Phantom/Vandal ties.",
    )
    p.add_argument(
        "--min-weapons-for-classify",
        type=int,
        default=1,
        metavar="N",
        help="With 2+ templates: require at least N weapons in the vote dict (after optional "
        "--supplement-class-scores). Default 1.",
    )
    p.add_argument(
        "--supplement-class-scores",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With 2+ templates: for weapons missing from peak votes, add global max NCC on the row "
        "search image so second place exists and --min-class-margin works on narrow white slots. "
        "--no-supplement-class-scores restores old behavior.",
    )
    p.add_argument(
        "--only-weapon",
        type=str,
        default=None,
        metavar="ID",
        help="Keep only rows whose winning template id equals this string (exact match, case-sensitive).",
    )
    p.add_argument(
        "--classify-radius-px",
        type=float,
        default=0.0,
        metavar="R",
        help="Radius for picking the box near the weapon anchor (0 = auto ~ max(28, 0.06·ROI width)).",
    )
    p.add_argument("--max-matches", type=int, default=8, help="Max detections after NMS (raise for debugging).")
    p.add_argument(
        "--peak-nms-frac",
        type=float,
        default=0.28,
        metavar="0..1",
        help="Per-scale peak suppression margin as fraction of max(template w,h).",
    )
    p.add_argument(
        "--merge-dist-frac",
        type=float,
        default=0.42,
        metavar="0..1",
        help="When merging across scales: min center distance = frac * max(box w,h) of the pair.",
    )
    p.add_argument(
        "--max-per-row",
        type=int,
        default=-1,
        metavar="N",
        help="After cross-scale merge, keep at most N hits per row patch (row bands or HSV rows). "
        "Default -1: use 1 when there are multiple patches, else no per-row cap. "
        "Use 0 to disable per-row capping (only --max-matches applies).",
    )
    p.add_argument(
        "--weapon-anchor-x-frac",
        type=float,
        default=0.62,
        metavar="0..1",
        help="With --max-per-row 1 and no white-weapon slot: NCC anchor X as fraction of strip width. "
        "Ignored when a white slot is detected (anchor = slot center).",
    )
    p.add_argument(
        "--classify-vote-min-x-frac",
        type=float,
        default=0.42,
        metavar="0..1",
        help="With --max-per-row 1 and **no** white-weapon slot: only NCC peaks with center X ≥ this "
        "fraction (from strip left) vote for weapon id. With white slot, a low internal threshold is used.",
    )
    p.add_argument(
        "--weapon-white-slot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With multiple row strips: detect a bright gun-shaped blob between names and match templates "
        "only in that X window (default: on). --no-weapon-white-slot searches the full center strip.",
    )
    p.add_argument(
        "--weapon-white-thr",
        type=int,
        default=205,
        metavar="0..255",
        help="Gray ≥ this value counts as white for weapon-slot detection (with --weapon-white-slot).",
    )
    p.add_argument(
        "--weapon-slot-pad",
        type=int,
        default=14,
        metavar="PX",
        help="Horizontal padding added around the detected white weapon blob before template matching.",
    )
    p.add_argument(
        "--weapon-slot-cx-min-frac",
        type=float,
        default=0.14,
        metavar="0..1",
        help="Reject white blobs whose center is left of this fraction of strip width (portrait / killer).",
    )
    p.add_argument(
        "--weapon-slot-cx-max-frac",
        type=float,
        default=0.86,
        metavar="0..1",
        help="Reject white blobs whose center is right of this fraction (victim name / edge).",
    )
    p.add_argument(
        "--scales",
        type=str,
        default="1.0",
        help="Comma-separated scale factors applied to template size, e.g. 0.85,1.0,1.15",
    )
    p.add_argument("--draw", type=Path, default=None, help="Write visualization (BGR) to this path.")
    p.add_argument(
        "--dump-roi",
        type=Path,
        default=None,
        help="Save the killfeed ROI actually searched (BGR) — use to verify crop vs full screenshot.",
    )
    p.add_argument("--quiet", action="store_true", help="Only exit code (0 if any hit).")
    p.add_argument(
        "--bench-repetitions",
        type=int,
        default=0,
        metavar="N",
        help="After image + template load: run 1 unmeasured warmup, then time N search+classify "
        "passes (white slot, matchTemplate, merge, classify) and print mean/min/max ms. 0 = off.",
    )
    args = p.parse_args()

    if args.template is None and args.templates_json is None:
        raise SystemExit("Provide either -t/--template or --templates-json.")
    if args.template is not None and args.templates_json is not None:
        raise SystemExit("Use either -t/--template or --templates-json, not both.")

    img_path = args.image.resolve()
    if not img_path.is_file():
        raise SystemExit(f"Not a file: {img_path}")

    full = cv2.imread(str(img_path))
    if full is None:
        raise SystemExit(f"Cannot read image: {img_path}")

    if args.killfeed_rect is None and not args.no_crop:
        roi = crop_killfeed_region_if_possible(full)
        h, w = full.shape[:2]
        x, y = REGION_KILLFEED["left"], REGION_KILLFEED["top"]
        rw, rh = REGION_KILLFEED["width"], REGION_KILLFEED["height"]
        offset_x, offset_y = (x, y) if (w >= x + rw and h >= y + rh) else (0, 0)
    else:
        roi, offset_x, offset_y = roi_from_image(
            full, killfeed_rect=args.killfeed_rect, no_crop=args.no_crop
        )

    if args.dump_roi is not None:
        args.dump_roi.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.dump_roi.resolve()), roi)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_h, roi_w = gray.shape[:2]

    row_boxes_for_draw: list[tuple[int, int, int, int]] = []
    patches: list[tuple[np.ndarray, int, int]] = []
    row_mode_active = False
    used_fixed_row_bands = False

    if args.row_bands_json is not None:
        rb_path = args.row_bands_json.resolve()
        if not rb_path.is_file():
            raise SystemExit(f"Not a file: {rb_path}")
        if args.within_kill_rows and not args.quiet:
            print("Note: --row-bands-json takes precedence over --within-kill-rows.")
        bands = load_row_bands_json(rb_path)
        patches, row_boxes_for_draw = patches_from_row_bands_frac(
            gray, bands, args.center_frac, roi_w, roi_h
        )
        row_mode_active = True
        used_fixed_row_bands = True
    elif args.within_kill_rows:
        raw_rows, _ = detect_killfeed_row_boxes(roi)
        split_rows = split_tall_killfeed_row_boxes(raw_rows)
        mcy = args.max_row_center_y_frac
        y_cutoff = None if mcy <= 0 else mcy
        filtered = filter_plausible_row_boxes(
            split_rows,
            roi_w,
            roi_h,
            min_width_frac=args.min_row_width_frac,
            max_center_y_frac=y_cutoff,
        )
        pad = max(0, args.row_pad)
        if filtered:
            row_mode_active = True
            for rx, ry, rw, rh, _c in filtered:
                x0, y0, bw, bh = clamp_row_box(rx, ry, rw, rh, pad, roi_w, roi_h)
                row_boxes_for_draw.append((x0, y0, bw, bh))
                sub = gray[y0 : y0 + bh, x0 : x0 + bw]
                sg, ix = center_strip(sub, args.center_frac)
                patches.append((sg, x0 + ix, y0))
        else:
            if not args.quiet:
                print(
                    "within-kill-rows: no row boxes passed width/y heuristics "
                    "(HSV often tags terrain in the lower strip, not the kill line). "
                    "Falling back to full ROI; use --center-frac to limit false matches."
                )
            sg, ix = center_strip(gray, args.center_frac)
            patches.append((sg, ix, 0))
    else:
        sg, ix = center_strip(gray, args.center_frac)
        patches.append((sg, ix, 0))

    if args.templates_json is not None:
        templates = load_templates_json(args.templates_json.resolve())
    else:
        assert args.template is not None
        tp = args.template.resolve()
        templates = {tp.stem: load_template(tp)}
    multitpl = len(templates) > 1
    scales = [float(s.strip()) for s in args.scales.split(",") if s.strip()]

    classify_r = float(args.classify_radius_px)
    if classify_r <= 0:
        classify_r = max(28.0, 0.06 * float(roi_w))

    only_w = args.only_weapon.strip() if args.only_weapon else None
    weapon_anchor_frac = max(0.35, min(0.85, float(args.weapon_anchor_x_frac)))
    vote_min_x_frac = float(args.classify_vote_min_x_frac)

    def classify_row_patch(
        patch_hits: list[HitTagged],
        *,
        base_x: int,
        strip_w: int,
        anchor_x: int,
        anchor_y: int,
        vote_x_frac: float,
        ncc_radius: float | None = None,
        match_gray: np.ndarray | None = None,
        supplement_gray: np.ndarray | None = None,
    ) -> list[tuple[str, int, int, int, int, float, float, float, str]]:
        """Weapon id from NCC peaks; vote strip uses vote_x_frac from strip left."""
        r = classify_r if ncc_radius is None else ncc_radius
        out: list[tuple[str, int, int, int, int, float, float, float, str]] = []
        vxf = max(0.0, min(0.85, float(vote_x_frac)))
        by_w = max_scores_by_weapon_right_of_strip(
            patch_hits,
            base_x=base_x,
            strip_w=strip_w,
            min_center_x_frac=vxf,
        )
        if not by_w:
            by_w = max_scores_by_weapon_near(patch_hits, anchor_x, anchor_y, r)
        if not by_w:
            return out
        peaks_by_w = dict(by_w)
        if multitpl and match_gray is not None and args.supplement_class_scores:
            gray_sup = supplement_gray if supplement_gray is not None else match_gray
            supplement_class_scores_from_gray(by_w, gray_sup, templates, scales)
        if multitpl and len(by_w) < max(1, int(args.min_weapons_for_classify)):
            return out
        if multitpl:
            # Winner only from real template peaks (slot/strip search). Supplement fills other classes
            # so margin reflects discrimination; otherwise unknown guns + narrow slot → fake margin.
            win_w, win_s = max(peaks_by_w.items(), key=lambda kv: kv[1])
            others = [wid for wid in templates if wid != win_w]
            if others:
                sec_w, sec_s = max(((w, by_w.get(w, 0.0)) for w in others), key=lambda kv: kv[1])
            else:
                sec_w, sec_s = "", 0.0
            margin = (win_s - sec_s) if sec_w else 0.0
        else:
            ranked = sorted(by_w.items(), key=lambda kv: kv[1], reverse=True)
            win_w, win_s = ranked[0]
            sec_w, sec_s = ranked[1] if len(ranked) > 1 else ("", 0.0)
            margin = (win_s - sec_s) if sec_w else 0.0

        if win_s < args.min_score:
            return out
        if multitpl and args.min_class_margin > 0:
            need_m = float(args.min_class_margin)
            if margin < need_m:
                # Rifle silhouettes (Vandal vs Phantom) often sit within ~0.001 NCC when both match
                # the same white icon crop; allow a tiny slack only for very strong winners.
                if not (
                    win_s >= 0.982
                    and margin >= need_m - 0.00085
                ):
                    return out
        if only_w is not None and win_w != only_w:
            return out

        geom = best_geometry_for_weapon(patch_hits, win_w, anchor_x, anchor_y, r)
        if geom is None:
            x_cut = base_x + int(round(strip_w * max(0.15, min(0.85, vxf))))
            right_only = [
                h for h in patch_hits if h[0] == win_w and h[1] + h[3] // 2 >= x_cut
            ]
            geom = max(right_only, key=lambda h: h[5]) if right_only else None
        if geom is None:
            same_id = [h for h in patch_hits if h[0] == win_w]
            geom = max(same_id, key=lambda h: h[5]) if same_id else None
        if geom is None:
            return out
        w_g, x, y, tw, th, sc_geom, sc_fn = geom
        out.append((w_g, x, y, tw, th, sc_geom, sc_fn, margin, sec_w))
        return out

    if args.max_per_row == -1:
        per_row_cap: int | None = 1 if len(patches) > 1 else None
    elif args.max_per_row == 0:
        per_row_cap = None
    else:
        per_row_cap = max(1, args.max_per_row)

    use_white_slot = bool(args.weapon_white_slot) and len(patches) > 1

    def run_match_and_classify() -> tuple[
        list[tuple[str, int, int, int, int, float, float, float, str]],
        list[tuple[int, int, int, int] | None],
    ]:
        merged_rows: list[tuple[str, int, int, int, int, float, float, float, str]] = []
        slot_boxes_roi: list[tuple[int, int, int, int] | None] = []
        if per_row_cap is not None:
            inner_cap = max(args.max_matches, per_row_cap * 8, 32)
            for search_gray, base_x, base_y in patches:
                strip_w, strip_h = search_gray.shape[1], search_gray.shape[0]
                slot_slice: tuple[int, int] | None = None
                if use_white_slot:
                    Hs, Ws = strip_h, strip_w
                    min_a = max(35, int(Hs * Ws * 0.0015))
                    max_a = min(12000, int(Hs * Ws * 0.12))
                    slot_slice = white_weapon_slot_x_slice(
                        search_gray,
                        white_thr=args.weapon_white_thr,
                        cx_min_frac=args.weapon_slot_cx_min_frac,
                        cx_max_frac=args.weapon_slot_cx_max_frac,
                        min_area=min_a,
                        max_area=max_a,
                        min_aspect=1.12,
                        max_aspect=7.5,
                        min_h_frac=0.26,
                        max_h_frac=0.99,
                        pad_px=max(0, args.weapon_slot_pad),
                    )
                if slot_slice is not None:
                    sx0, sx1 = slot_slice
                    min_tw, _min_th = min_slot_dims_for_all_templates(templates, scales)
                    if strip_w >= min_tw:
                        sx0, sx1 = expand_x_slice_to_min_width(sx0, sx1, strip_w, min_tw)
                    match_gray = search_gray[:, sx0:sx1]
                    eff_base_x = base_x + sx0
                    eff_w = sx1 - sx0
                    anchor_x = eff_base_x + eff_w // 2
                    vote_xf = 0.06
                    # Anchor is strip center; classify_r (from full ROI) can be smaller than half the
                    # match crop, so no peak lies inside the radius and geometry lookup fails.
                    half_diag = math.hypot(float(eff_w) * 0.5, float(strip_h) * 0.5)
                    local_r = max(
                        14.0,
                        half_diag * 0.98,
                        min(float(classify_r), 0.25 * float(strip_w)),
                    )
                    slot_boxes_roi.append(
                        (eff_base_x, base_y, eff_base_x + eff_w - 1, base_y + strip_h - 1)
                    )
                else:
                    match_gray = search_gray
                    eff_base_x = base_x
                    eff_w = strip_w
                    anchor_x = base_x + int(round(strip_w * weapon_anchor_frac))
                    vote_xf = vote_min_x_frac
                    local_r = None
                    slot_boxes_roi.append(None)
                    if use_white_slot and not args.quiet:
                        print(
                            f"  row y={base_y}: white weapon slot not found, using full center strip"
                        )

                anchor_y = base_y + strip_h // 2
                patch_hits: list[HitTagged] = []
                for wname, (templ, mask) in templates.items():
                    for sc in scales:
                        for x, y, tw, th, score, sc2 in match_at_scale(
                            match_gray,
                            templ,
                            mask,
                            sc,
                            min_score=args.min_score,
                            max_matches=inner_cap,
                            min_dist_frac=args.peak_nms_frac,
                        ):
                            patch_hits.append(
                                (wname, eff_base_x + x, base_y + y, tw, th, score, sc2)
                            )
                if per_row_cap == 1:
                    merged_rows.extend(
                        classify_row_patch(
                            patch_hits,
                            base_x=eff_base_x,
                            strip_w=eff_w,
                            anchor_x=anchor_x,
                            anchor_y=anchor_y,
                            vote_x_frac=vote_xf,
                            ncc_radius=local_r,
                            match_gray=match_gray,
                            supplement_gray=search_gray,
                        )
                    )
                else:
                    row_merged = merge_tagged_hits_across_scales(
                        patch_hits,
                        merge_dist_frac=args.merge_dist_frac,
                        max_matches=inner_cap,
                    )
                    row_merged.sort(key=lambda t: t[5], reverse=True)
                    for cand in row_merged[:per_row_cap]:
                        ax = cand[1] + cand[3] // 2
                        ay = cand[2] + cand[4] // 2
                        merged_rows.extend(
                            classify_row_patch(
                                patch_hits,
                                base_x=eff_base_x,
                                strip_w=eff_w,
                                anchor_x=ax,
                                anchor_y=ay,
                                vote_x_frac=vote_xf,
                                ncc_radius=local_r,
                                match_gray=match_gray,
                                supplement_gray=search_gray,
                            )
                        )
            merged_rows.sort(key=lambda t: t[5], reverse=True)
            merged_rows = merged_rows[: args.max_matches]
        else:
            all_hits: list[HitTagged] = []
            for search_gray, base_x, base_y in patches:
                for wname, (templ, mask) in templates.items():
                    for sc in scales:
                        for x, y, tw, th, score, sc2 in match_at_scale(
                            search_gray,
                            templ,
                            mask,
                            sc,
                            min_score=args.min_score,
                            max_matches=args.max_matches,
                            min_dist_frac=args.peak_nms_frac,
                        ):
                            all_hits.append((wname, base_x + x, base_y + y, tw, th, score, sc2))
            row_merged = merge_tagged_hits_across_scales(
                all_hits,
                merge_dist_frac=args.merge_dist_frac,
                max_matches=args.max_matches,
            )
            merged_rows.clear()
            for cand in row_merged:
                ax = cand[1] + cand[3] // 2
                ay = cand[2] + cand[4] // 2
                merged_rows.extend(
                    classify_row_patch(
                        all_hits,
                        base_x=0,
                        strip_w=roi_w,
                        anchor_x=ax,
                        anchor_y=ay,
                        vote_x_frac=vote_min_x_frac,
                    )
                )
            merged_rows.sort(key=lambda t: t[5], reverse=True)
            del merged_rows[args.max_matches :]
        return merged_rows, slot_boxes_roi

    bench_n = max(0, int(args.bench_repetitions))
    if bench_n > 0:
        run_match_and_classify()
        bench_ms: list[float] = []
        for _ in range(bench_n):
            t0 = time.perf_counter()
            merged_rows, slot_boxes_roi = run_match_and_classify()
            bench_ms.append((time.perf_counter() - t0) * 1000.0)
    else:
        merged_rows, slot_boxes_roi = run_match_and_classify()

    if not args.quiet:
        print(f"Image: {img_path}")
        print(f"ROI size: {roi.shape[1]}x{roi.shape[0]} px (offset in full image: {offset_x},{offset_y})")
        if args.templates_json is not None:
            print(f"Templates: {args.templates_json} ({len(templates)} ids, scales={scales})")
        else:
            assert args.template is not None
            t0, m0 = next(iter(templates.values()))
            print(f"Template: {args.template} ({t0.shape[1]}x{t0.shape[0]} px, scales={scales})")
        if per_row_cap == 1:
            cmsg = (
                f"Classify: white_slot={'on' if use_white_slot else 'off'}, "
                f"vote_min_x_frac={vote_min_x_frac:.2f} (used if slot off), "
                f"weapon_anchor_x_frac={weapon_anchor_frac:.2f} (if slot off), radius={classify_r:.1f} px"
            )
            if multitpl:
                cmsg += f", min_class_margin={args.min_class_margin:.3f}"
            if only_w:
                cmsg += f", only_weapon={only_w!r}"
            print(cmsg)
        elif multitpl:
            print(
                f"Classify: radius={classify_r:.1f} px, min_class_margin={args.min_class_margin:.3f}"
                + (f", only_weapon={only_w!r}" if only_w else "")
            )
        if used_fixed_row_bands:
            print(f"Row bands (JSON): {len(patches)} strip(s) from {args.row_bands_json}")
        elif args.within_kill_rows:
            if row_mode_active:
                print(
                    f"Within-kill-rows: {len(patches)} row patch(es), row_pad={args.row_pad}, "
                    f"min_row_w>={max(120, int(roi_w * max(0.15, min(0.95, args.min_row_width_frac))))} px"
                )
            else:
                print("Within-kill-rows: using full ROI (fallback after filters).")
        if merged_rows:
            print(f"Hits ({len(merged_rows)}):")
            for i, (wid, x, y, tw, th, score, sc, margin, sec_w) in enumerate(merged_rows, 1):
                fx = offset_x + x
                fy = offset_y + y
                extra = ""
                if multitpl:
                    extra = f" margin={margin:.3f}"
                    if sec_w:
                        extra += f" vs_{sec_w}={score - margin:.3f}"
                print(
                    f"  #{i} weapon={wid} score={score:.3f} scale={sc:.3f}{extra} "
                    f"roi=({x},{y}) size={tw}x{th} full_image=({fx},{fy})"
                )
        else:
            print(
                "No hits. Try lowering --min-score, adding scales, or (multi-template) "
                "lowering --min-class-margin / adding missing weapon templates."
            )

    if bench_n > 0:
        avg = sum(bench_ms) / len(bench_ms)
        print(
            f"Search+classify (1 warmup, {bench_n} timed): "
            f"mean {avg:.2f} ms, min {min(bench_ms):.2f} ms, max {max(bench_ms):.2f} ms"
        )

    if args.draw is not None:
        vis = roi.copy()
        if (used_fixed_row_bands or args.within_kill_rows) and row_boxes_for_draw:
            for x0, y0, bw, bh in row_boxes_for_draw:
                cv2.rectangle(vis, (x0, y0), (x0 + bw, y0 + bh), (0, 165, 255), 1)
        if len(slot_boxes_roi) == len(patches):
            for sb in slot_boxes_roi:
                if sb is not None:
                    sx0, sy0, sx1, sy1 = sb
                    cv2.rectangle(vis, (sx0, sy0), (sx1, sy1), (255, 0, 255), 1)
        for wid, x, y, tw, th, score, sc, margin, sec_w in merged_rows:
            cv2.rectangle(vis, (x, y), (x + tw, y + th), (0, 255, 0), 1)
            label = f"{wid} {score:.2f}"
            if multitpl and sec_w:
                label += f" m{margin:.2f}"
            cv2.putText(
                vis,
                label,
                (x, max(12, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        args.draw.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.draw.resolve()), vis)

    raise SystemExit(0 if merged_rows else 2)


if __name__ == "__main__":
    main()
