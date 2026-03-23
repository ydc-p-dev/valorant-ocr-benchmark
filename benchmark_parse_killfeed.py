#!/usr/bin/env python3
"""
Benchmark full unified killfeed parse: row bands + masks, optional weapon templates, per-row OCR.

Uses :func:`parse_killfeed.parse_killfeed_roi_unified` (same path as ``parse_killfeed.py``).

Examples:
  python benchmark_parse_killfeed.py --folder killfeed_screenshots --repeats 5 --warmup 1
  python benchmark_parse_killfeed.py --images shot.png --ocr-engine tesseract --no-weapons
  python benchmark_parse_killfeed.py --folder shots --cpu --out benchmark_parse_results.json
"""
from __future__ import annotations

import os
import sys

if "--cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
import statistics
import time
from typing import Any

import valorant_killfeed_tracker as vkt
from parse_killfeed import parse_killfeed_roi_unified
from scripts.match_killfeed_weapon import WeaponMatchParams


def collect_images(folder: Path | None, files: list[Path]) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    out: list[Path] = []
    if folder is not None:
        out.extend(sorted(p for p in folder.iterdir() if p.suffix.lower() in exts))
    out.extend(files)
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def _clear_row_cache() -> None:
    vkt._row_ocr_cache.clear()


def run_dataset_once(
    paths: list[Path],
    *,
    bands: list[tuple[float, float]],
    templates_json: Path | None,
    weapon_params: WeaponMatchParams,
    ocr_engine: str,
    highlight_min_abs: int,
    highlight_min_frac: float,
) -> list[dict[str, Any]]:
    now = time.time()
    timings: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for p in paths:
        full = vkt.load_bgr(p)
        roi = vkt.crop_killfeed_region_if_possible(full)
        events, _boxes, _masks, _band_active, _wh = parse_killfeed_roi_unified(
            roi,
            bands,
            templates_json=templates_json,
            weapon_params=weapon_params,
            ocr_engine=ocr_engine,
            now=now,
            highlight_min_abs=highlight_min_abs,
            highlight_min_frac=highlight_min_frac,
            timings_out=timings,
        )
        rows.append(
            {
                "image": p.name,
                "path": str(p.resolve()),
                "total_ms": float(timings["total_ms"]),
                "setup_ms": float(timings["setup_ms"]),
                "weapons_ms": float(timings["weapons_ms"]),
                "ocr_ms": float(timings["ocr_ms"]),
                "n_bands": len(events),
                "n_active": sum(1 for e in events if e.active),
                "n_with_weapon": sum(1 for e in events if e.active and e.weapon),
            }
        )
    return rows


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark unified killfeed parse (OCR + optional weapons, row bands)."
    )
    parser.add_argument("--folder", type=str, default=None, help="Directory of images.")
    parser.add_argument(
        "--images",
        nargs="*",
        default=[],
        help="Extra image paths (in addition to --folder).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        metavar="N",
        help="Use only first N images after sort.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force EasyOCR/PyTorch CPU (CUDA_VISIBLE_DEVICES=-1 before imports).",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Full dataset passes before timing.")
    parser.add_argument("--repeats", type=int, default=5, help="Timed passes (median per metric).")
    parser.add_argument(
        "--clear-cache-each-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear row OCR LRU before each timed pass (default: on).",
    )
    parser.add_argument(
        "--row-bands-json",
        type=Path,
        default=ROOT / "config" / "killfeed_row_bands.json",
        help="row_bands_frac JSON.",
    )
    parser.add_argument(
        "--weapon-templates-json",
        type=Path,
        default=ROOT / "config" / "weapon_templates.json",
        help="Weapon templates JSON. Use with --no-weapons to disable.",
    )
    parser.add_argument("--no-weapons", action="store_true", help="Skip weapon template matching.")
    parser.add_argument(
        "--ocr-engine",
        type=str,
        choices=["tesseract", "easyocr", "both"],
        default=vkt.DEFAULT_OCR_ENGINE,
    )
    parser.add_argument(
        "--killfeed-rect",
        type=str,
        default=None,
        metavar="TOP,LEFT,WIDTH,HEIGHT",
        help="Override killfeed rect for cropping full screenshots.",
    )
    parser.add_argument(
        "--min-band-highlight-px",
        type=int,
        default=280,
        metavar="N",
        help="Min HSV highlight pixels for an active row band.",
    )
    parser.add_argument(
        "--min-band-highlight-frac",
        type=float,
        default=0.001,
        metavar="0..1",
        help="Min highlight fraction of band area.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmark_parse_killfeed_results.json",
        help="Write full results JSON.",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optional per-image summary CSV.")
    args = parser.parse_args()

    if args.cpu:
        print("Benchmark: CPU-only (CUDA disabled for this process).")

    if args.killfeed_rect:
        parts = [int(x.strip()) for x in args.killfeed_rect.split(",")]
        if len(parts) != 4 or any(x < 0 for x in parts):
            raise SystemExit("--killfeed-rect: need TOP,LEFT,WIDTH,HEIGHT")
        vkt.REGION_KILLFEED = {
            "top": parts[0],
            "left": parts[1],
            "width": parts[2],
            "height": parts[3],
        }

    bands_path = args.row_bands_json.resolve()
    if not bands_path.is_file():
        raise SystemExit(f"Not a file: {bands_path}")
    bands = vkt.load_row_bands_json(bands_path)

    wjson = None if args.no_weapons else args.weapon_templates_json.resolve()
    if wjson is not None and not wjson.is_file():
        raise SystemExit(f"Weapon templates not found: {wjson} (use --no-weapons)")

    paths = collect_images(
        Path(args.folder) if args.folder else None,
        [Path(p) for p in args.images],
    )
    if not paths:
        raise SystemExit("No images: use --folder and/or --images.")
    if args.max_images is not None:
        paths = paths[: max(0, args.max_images)]

    print(f"Images ({len(paths)}): {', '.join(p.name for p in paths)}")
    print(
        f"Config: ocr_engine={args.ocr_engine!r}  weapons={'off' if args.no_weapons else str(wjson)}  "
        f"bands={bands_path.name}"
    )

    vkt.warm_easyocr_for_session(args.ocr_engine)
    wp = WeaponMatchParams()

    run_kw = dict(
        bands=bands,
        templates_json=wjson,
        weapon_params=wp,
        ocr_engine=args.ocr_engine,
        highlight_min_abs=args.min_band_highlight_px,
        highlight_min_frac=args.min_band_highlight_frac,
    )

    for _ in range(max(0, args.warmup)):
        _clear_row_cache()
        run_dataset_once(paths, **run_kw)

    repeat_rows: list[list[dict[str, Any]]] = []
    for _ in range(max(1, args.repeats)):
        if args.clear_cache_each_pass:
            _clear_row_cache()
        repeat_rows.append(run_dataset_once(paths, **run_kw))

    by_image: dict[str, list[dict[str, Any]]] = {p.name: [] for p in paths}
    for run in repeat_rows:
        for row in run:
            by_image[row["image"]].append(row)

    per_image_out: list[dict[str, Any]] = []
    for name in (p.name for p in paths):
        samples = by_image[name]
        totals = [float(s["total_ms"]) for s in samples]
        per_image_out.append(
            {
                "image": name,
                "path": samples[0]["path"] if samples else "",
                "median_total_ms": _median(totals),
                "mean_total_ms": float(statistics.mean(totals)) if totals else 0.0,
                "min_total_ms": float(min(totals)) if totals else 0.0,
                "max_total_ms": float(max(totals)) if totals else 0.0,
                "stdev_total_ms": float(statistics.stdev(totals)) if len(totals) > 1 else 0.0,
                "median_setup_ms": _median([float(s["setup_ms"]) for s in samples]),
                "median_weapons_ms": _median([float(s["weapons_ms"]) for s in samples]),
                "median_ocr_ms": _median([float(s["ocr_ms"]) for s in samples]),
                "n_bands": int(samples[-1]["n_bands"]) if samples else 0,
                "n_active": int(samples[-1]["n_active"]) if samples else 0,
                "n_with_weapon": int(samples[-1]["n_with_weapon"]) if samples else 0,
                "repeats": len(samples),
            }
        )

    sum_per_repeat = [
        sum(float(r["total_ms"]) for r in run) for run in repeat_rows
    ]
    setup_per_repeat = [sum(float(r["setup_ms"]) for r in run) for run in repeat_rows]
    weapons_per_repeat = [sum(float(r["weapons_ms"]) for r in run) for run in repeat_rows]
    ocr_per_repeat = [sum(float(r["ocr_ms"]) for r in run) for run in repeat_rows]

    payload = {
        "parser": "parse_killfeed_roi_unified",
        "images": [str(p) for p in paths],
        "row_bands_json": str(bands_path),
        "weapon_templates_json": None if wjson is None else str(wjson),
        "ocr_engine": args.ocr_engine,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "clear_cache_each_pass": args.clear_cache_each_pass,
        "cpu_only": bool(args.cpu),
        "highlight_min_abs": args.min_band_highlight_px,
        "highlight_min_frac": args.min_band_highlight_frac,
        "dataset": {
            "median_sum_total_ms": _median(sum_per_repeat),
            "median_sum_setup_ms": _median(setup_per_repeat),
            "median_sum_weapons_ms": _median(weapons_per_repeat),
            "median_sum_ocr_ms": _median(ocr_per_repeat),
            "median_total_ms_per_image": _median(sum_per_repeat) / max(len(paths), 1),
        },
        "per_image": per_image_out,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "image",
            "median_total_ms",
            "mean_total_ms",
            "min_total_ms",
            "max_total_ms",
            "stdev_total_ms",
            "median_setup_ms",
            "median_weapons_ms",
            "median_ocr_ms",
            "n_bands",
            "n_active",
            "n_with_weapon",
        ]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in per_image_out:
                w.writerow({k: row[k] for k in fields})

    d = payload["dataset"]
    print(f"Wrote {args.out.resolve()}")
    print(
        f"Dataset (median over {args.repeats} pass(es)): "
        f"sum_total={d['median_sum_total_ms']:.1f} ms  "
        f"per_image~={d['median_total_ms_per_image']:.1f} ms  "
        f"(setup {d['median_sum_setup_ms']:.1f} + weapons {d['median_sum_weapons_ms']:.1f} + ocr {d['median_sum_ocr_ms']:.1f})"
    )
    print("Per image (median total_ms):")
    for row in per_image_out:
        print(
            f"  {row['image']}: {row['median_total_ms']:.1f} ms  "
            f"[setup {row['median_setup_ms']:.1f}  weapons {row['median_weapons_ms']:.1f}  ocr {row['median_ocr_ms']:.1f}]  "
            f"active {row['n_active']}/{row['n_bands']}  weapon_hits {row['n_with_weapon']}"
        )
    if args.csv:
        print(f"CSV -> {args.csv.resolve()}")


if __name__ == "__main__":
    main()
