#!/usr/bin/env python3
"""
Normalize a wiki/fandom weapon icon into a killfeed-style silhouette for template matching.

Example (Vandal, barrel points left in source → flip to match in-game killfeed):

  python scripts/process_weapon_icon.py wiki_icon.png -o assets/icons/MyWeapon_killfeed.png

Tune thresholds on your assets if the mask is too thin or too noisy:
  --alpha-thresh, --gray-thresh, --mask-and, --close-iter, --dilate-iter, --no-flip

Transparent PNGs: mask is mostly from alpha (BGR under transparent pixels is ignored).
Opaque icons: mask uses grayscale threshold only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def load_bgra(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise SystemExit(f"Cannot read image: {path}")
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
    elif im.shape[2] == 3:
        bgra = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = 255
        im = bgra
    return im


def build_mask(
    bgra: np.ndarray,
    *,
    alpha_thresh: int,
    gray_thresh: int,
    prefer_alpha_or_gray: bool,
) -> np.ndarray:
    """Binary uint8 mask 0/255.

    Wiki/Fandom PNGs often keep non-zero BGR under fully transparent pixels (e.g. gray ~64).
    Then (alpha>=t) OR (gray>=low) marks the whole canvas. If a meaningful share of pixels
    is below ``alpha_thresh``, we treat the asset as true RGBA and use alpha as the primary
    ink signal; otherwise we use grayscale only (opaque icons).
    """
    bgr = bgra[:, :, :3]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    a = bgra[:, :, 3]

    at = int(np.clip(alpha_thresh, 0, 255))
    transparent_frac = float(np.mean(a < at))
    m_a = a >= at
    m_g = gray >= gray_thresh

    if transparent_frac > 0.05:
        mask = m_a
        if not prefer_alpha_or_gray:
            mask = np.logical_and(mask, m_g)
    else:
        mask = m_g
        if not prefer_alpha_or_gray:
            mask = np.logical_and(mask, m_a)
    return (mask.astype(np.uint8) * 255)


def trim_bbox(mask: np.ndarray, pad: int) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return mask
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    h, w = mask.shape[:2]
    y0 = max(0, y0 - pad)
    y1 = min(h, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(w, x1 + pad)
    return mask[y0:y1, x0:x1]


def resize_to_height(mask: np.ndarray, target_h: int) -> np.ndarray:
    if target_h <= 0:
        return mask
    h, w = mask.shape[:2]
    if h == 0:
        return mask
    new_w = max(1, int(round(w * target_h / h)))
    return cv2.resize(mask, (new_w, target_h), interpolation=cv2.INTER_AREA)


def mask_to_bgra_white(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[:, :, 0] = 255
    out[:, :, 1] = 255
    out[:, :, 2] = 255
    out[:, :, 3] = mask
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Weapon icon → killfeed-style silhouette template.")
    p.add_argument("input", type=Path, help="Input PNG/WebP path (RGBA preferred).")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (.png). Default: <input-dir>/processed/<stem>_template.png (e.g. -o assets/icons/Weapon_killfeed.png)",
    )
    p.add_argument("--height", type=int, default=28, help="Silhouette height in px (killfeed-scale).")
    p.add_argument("--no-flip", action="store_true", help="Do not mirror horizontally.")
    p.add_argument("--alpha-thresh", type=int, default=12, help="Min alpha to count as ink (0–255).")
    p.add_argument("--gray-thresh", type=int, default=35, help="Min grayscale (0–255) on BGR.")
    p.add_argument(
        "--mask-and",
        action="store_true",
        help="Stricter: AND grayscale with the alpha-based (transparent PNG) or gray-only (opaque) mask.",
    )
    p.add_argument("--close-iter", type=int, default=2, help="Morphology CLOSE iterations (fill gaps).")
    p.add_argument("--dilate-iter", type=int, default=0, help="Extra DILATE iterations (thicken silhouette).")
    p.add_argument(
        "--as-mask",
        action="store_true",
        help="Save single-channel mask instead of white-on-transparent BGRA.",
    )
    p.add_argument("--crop-pad", type=int, default=2, help="Padding px around bbox trim.")
    args = p.parse_args()

    inp = args.input.resolve()
    if not inp.is_file():
        raise SystemExit(f"Not a file: {inp}")

    out = args.output
    if out is None:
        proc = inp.parent / "processed"
        proc.mkdir(parents=True, exist_ok=True)
        out = proc / f"{inp.stem}_template.png"
    else:
        out = args.output.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

    bgra = load_bgra(inp)
    if not args.no_flip:
        bgra = cv2.flip(bgra, 1)

    mask = build_mask(
        bgra,
        alpha_thresh=args.alpha_thresh,
        gray_thresh=args.gray_thresh,
        prefer_alpha_or_gray=not args.mask_and,
    )

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if args.close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=args.close_iter)
    if args.dilate_iter > 0:
        mask = cv2.dilate(mask, k, iterations=args.dilate_iter)

    mask = trim_bbox(mask, args.crop_pad)
    mask = resize_to_height(mask, args.height)

    if args.as_mask:
        cv2.imwrite(str(out), mask)
    else:
        cv2.imwrite(str(out), mask_to_bgra_white(mask))

    print(f"Wrote {out} ({mask.shape[1]}x{mask.shape[0]} px)")


if __name__ == "__main__":
    main()
