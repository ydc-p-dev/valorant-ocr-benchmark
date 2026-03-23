#!/usr/bin/env python3
"""
Horizontally flip image(s). Preserves alpha (BGRA) and single-channel masks.

Examples:

  python scripts/flip_horizontal.py assets/icons/Phantom_killfeed.png
  # writes assets/icons/Phantom_killfeed_hflip.png

  python scripts/flip_horizontal.py a.png b.png --in-place
  python scripts/flip_horizontal.py icon.png -o assets/icon_flipped.png
  python scripts/flip_horizontal.py assets/foo.png assets/bar.png --out-dir assets/flipped
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    p = argparse.ArgumentParser(description="Flip images horizontally (mirror left/right).")
    p.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Input image path(s) (png, webp, etc.).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output path (only when exactly one input file).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Write outputs here, same filenames as inputs (directory is created if missing).",
    )
    p.add_argument(
        "--suffix",
        type=str,
        default="_hflip",
        help="When saving next to input: stem{suffix}.ext (default: %(default)s).",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite each input file.",
    )
    args = p.parse_args()

    inputs = [x.resolve() for x in args.images]
    for inp in inputs:
        if not inp.is_file():
            raise SystemExit(f"Not a file: {inp}")

    if args.in_place and (args.output is not None or args.out_dir is not None):
        raise SystemExit("Use --in-place alone, or use -o / --out-dir (not together).")
    if args.output is not None and len(inputs) != 1:
        raise SystemExit("-o/--output requires exactly one input image.")
    if args.out_dir is not None and args.output is not None:
        raise SystemExit("Use either -o or --out-dir, not both.")

    if args.out_dir is not None:
        out_dir = args.out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    for inp in inputs:
        im = cv2.imread(str(inp), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise SystemExit(f"Cannot read: {inp}")
        flipped = cv2.flip(im, 1)

        if args.in_place:
            out_path = inp
        elif args.output is not None:
            out_path = args.output.resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
        elif args.out_dir is not None:
            out_path = out_dir / inp.name
        else:
            out_path = inp.with_name(f"{inp.stem}{args.suffix}{inp.suffix}")

        if not cv2.imwrite(str(out_path), flipped):
            raise SystemExit(f"Cannot write: {out_path}")
        print(out_path)


if __name__ == "__main__":
    main()
