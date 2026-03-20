#!/usr/bin/env python3
"""
Grid benchmark: sweep EasyOCR (and optionally Tesseract) on static killfeed images.

Examples:
  python benchmark_killfeed_ocr.py --folder killfeed_screenshots --repeats 5 --warmup 1
  python benchmark_killfeed_ocr.py --images shot.png --canvas-sizes 480 640
  python benchmark_killfeed_ocr.py --folder shots --networks craft dbnet18 --force-dbnet-sweep
  python benchmark_killfeed_ocr.py --images a.png b.png --cpu --engines easyocr  # EasyOCR on CPU only
"""
from __future__ import annotations

import os
import sys

# Before PyTorch/EasyOCR import: optional CPU-only (hides CUDA devices).
if "--cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Import torch only after this (via valorant_killfeed_tracker): cuts some PyTorch C++ extension noise.
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import argparse
import csv
import json
import shutil
import statistics
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import valorant_killfeed_tracker as vkt


@dataclass(frozen=True)
class EasyOCRGridConfig:
    detect_network: str
    canvas_size: int
    stack_rows: bool


def _clear_row_cache() -> None:
    vkt._row_ocr_cache.clear()


def _apply_easyocr(cfg: EasyOCRGridConfig) -> None:
    vkt._easyocr_detect_network = cfg.detect_network
    vkt.EASYOCR_CANVAS_SIZE = max(64, min(4096, int(cfg.canvas_size)))
    vkt.EASYOCR_STACK_ROWS = bool(cfg.stack_rows)
    vkt._easyocr_invalidate_reader()
    _clear_row_cache()


def _windows_dbnet_toolchain_unlikely() -> bool:
    """DBNet JIT needs MSVC (cl) and CUDA_HOME on Windows; without them EasyOCR spams and always falls back."""
    if sys.platform != "win32":
        return False
    if not os.environ.get("CUDA_HOME", "").strip():
        return True
    if not shutil.which("cl"):
        return True
    return False


def _filter_networks_for_platform(
    networks: list[str],
    *,
    force_dbnet: bool,
) -> tuple[list[str], str | None]:
    """Returns (networks, skip_note)."""
    if "dbnet18" not in networks:
        return networks, None
    if force_dbnet or not _windows_dbnet_toolchain_unlikely():
        return networks, None
    kept = [n for n in networks if n != "dbnet18"]
    if not kept:
        kept = ["craft"]
    note = (
        "Skipping EasyOCR dbnet18 on this Windows setup (no CUDA_HOME or MSVC cl). "
        "Sweeps duplicate craft vs fallback craft. Use --force-dbnet-sweep to include dbnet18 anyway."
    )
    return kept, note


def _events_signature(events: list[vkt.KillfeedEvent]) -> list[tuple[str, str]]:
    return [(e.killer.strip().lower(), e.victim.strip().lower()) for e in events]


def _completeness_ratio(events: list[vkt.KillfeedEvent], rows_detected: int) -> float:
    if rows_detected <= 0:
        return 1.0 if not events else 0.0
    ok = sum(
        1
        for e in events
        if e.killer and e.killer != "?" and e.victim and e.victim != "?"
    )
    return min(1.0, ok / rows_detected)


def _reference_accuracy(
    image_name: str,
    events: list[vkt.KillfeedEvent],
    reference: dict[str, list[dict[str, str]]],
) -> float | None:
    ref = reference.get(image_name)
    if not ref:
        return None
    got = _events_signature(events)
    exp = [(r["killer"].strip().lower(), r["victim"].strip().lower()) for r in ref]
    if len(got) != len(exp):
        return 0.0
    if not exp:
        return 1.0
    hits = sum(1 for g, e in zip(got, exp) if g == e)
    return hits / len(exp)


def _run_dataset_once(
    paths: list[Path],
    ocr_engine: str,
) -> tuple[list[dict[str, Any]], float, float]:
    rows: list[dict[str, Any]] = []
    parse_sum = 0.0
    total_sum = 0.0
    now = time.time()
    for p in paths:
        frame = vkt.crop_killfeed_region_if_possible(vkt.load_bgr(p))
        events, _boxes, timing = vkt.process_frame(
            frame, now, recent_pairs=None, draw=None, ocr_engine=ocr_engine
        )
        parse_sum += float(timing["t_parse_ms_total"])
        total_sum += float(timing["t_total_ms"])
        rows.append(
            {
                "image": p.name,
                "events": events,
                "rows_detected": int(timing["rows_detected"]),
                "t_parse_ms": float(timing["t_parse_ms_total"]),
                "t_total_ms": float(timing["t_total_ms"]),
            }
        )
    return rows, parse_sum, total_sum


def _score_run(
    per_image: list[dict[str, Any]],
    reference: dict[str, list[dict[str, str]]] | None,
) -> dict[str, float | None]:
    comp = [
        _completeness_ratio(r["events"], r["rows_detected"]) for r in per_image
    ]
    mean_comp = float(statistics.mean(comp)) if comp else 0.0

    ref_scores: list[float] = []
    if reference:
        for r in per_image:
            acc = _reference_accuracy(r["image"], r["events"], reference)
            if acc is not None:
                ref_scores.append(acc)
    mean_ref = float(statistics.mean(ref_scores)) if ref_scores else None

    return {"mean_completeness": mean_comp, "mean_reference_accuracy": mean_ref}


def _balanced_rank(
    parse_ms_per_image: float,
    mean_comp: float,
    mean_ref: float | None,
    *,
    ref_weight: float,
) -> float:
    """Higher is better."""
    t = max(parse_ms_per_image, 1.0)
    speed = 1000.0 / (t**0.5)
    qual = mean_comp
    if mean_ref is not None:
        qual = (1.0 - ref_weight) * mean_comp + ref_weight * mean_ref
    return 0.45 * speed + 0.55 * qual * 200.0


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


def _serialize_last_run(per_image: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ser: list[dict[str, Any]] = []
    for r in per_image:
        ser.append(
            {
                "image": r["image"],
                "rows_detected": r["rows_detected"],
                "t_parse_ms": r["t_parse_ms"],
                "t_total_ms": r["t_total_ms"],
                "pairs": [
                    {"killer": e.killer, "victim": e.victim, "row_color": e.row_color}
                    for e in r["events"]
                ],
            }
        )
    return ser


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark killfeed OCR: grid over EasyOCR params (+ optional engines)."
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
        help="Use only first N images after sort (quick sweeps on large folders).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force EasyOCR/PyTorch CPU (sets CUDA_VISIBLE_DEVICES=-1 before import). Slower; for comparison vs GPU.",
    )
    parser.add_argument(
        "--engines",
        nargs="*",
        default=["easyocr"],
        choices=["easyocr", "tesseract", "both"],
        help="OCR engines to include.",
    )
    parser.add_argument(
        "--networks",
        nargs="*",
        default=["craft"],
        choices=["craft", "dbnet18"],
        help="EasyOCR detector sweep. Default: craft only (avoids DBNet JIT noise on Windows).",
    )
    parser.add_argument(
        "--force-dbnet-sweep",
        action="store_true",
        help="Include dbnet18 even when CUDA_HOME/MSVC are missing (loud, usually falls back to craft).",
    )
    parser.add_argument(
        "--canvas-sizes",
        nargs="*",
        type=int,
        default=[480, 640, 960],
        help="EasyOCR canvas_size sweep.",
    )
    parser.add_argument(
        "--stack-rows",
        nargs="*",
        type=str,
        default=["true", "false"],
        metavar="BOOL",
        help='Per-row stacked readtext: "true" or "false".',
    )
    parser.add_argument("--warmup", type=int, default=1, help="Full dataset passes per config before timing.")
    parser.add_argument("--repeats", type=int, default=5, help="Timed passes; median aggregate ms reported.")
    parser.add_argument(
        "--clear-cache-each-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear row OCR LRU before each timed pass (default: on). "
        "Use --no-clear-cache-each-pass to measure steady-state with hot cache.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help='JSON: { "img.png": [ {"killer":"a","victim":"b"}, ... ] }',
    )
    parser.add_argument(
        "--out",
        type=str,
        default="benchmark_killfeed_results.json",
        help="Full results JSON.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional summary CSV.",
    )
    parser.add_argument(
        "--ref-weight",
        type=float,
        default=0.6,
        help="With --reference: weight of reference accuracy vs completeness (0..1).",
    )
    args = parser.parse_args()

    if args.cpu:
        print("Benchmark: CPU-only (CUDA disabled for this process).")

    raw_networks = list(args.networks) if args.networks else ["craft"]
    net_list, skip_note = _filter_networks_for_platform(
        raw_networks,
        force_dbnet=args.force_dbnet_sweep,
    )
    if skip_note:
        print(skip_note, file=sys.stderr)

    paths = collect_images(
        Path(args.folder) if args.folder else None,
        [Path(p) for p in args.images],
    )
    if not paths:
        raise SystemExit("No images: use --folder and/or --images.")
    if args.max_images is not None:
        paths = paths[: max(0, args.max_images)]

    names_line = ", ".join(p.name for p in paths)
    print(f"Benchmark images ({len(paths)}): {names_line}")
    images_csv = ";".join(p.name for p in paths)

    reference: dict[str, list[dict[str, str]]] | None = None
    if args.reference:
        with open(args.reference, encoding="utf-8") as f:
            reference = json.load(f)

    if "easyocr" in args.engines or "both" in args.engines:
        if vkt.easyocr is None:
            raise SystemExit("EasyOCR not installed.")

    stack_flags: list[bool] = []
    for s in args.stack_rows:
        sl = s.lower()
        if sl in ("1", "true", "yes", "y"):
            stack_flags.append(True)
        elif sl in ("0", "false", "no", "n"):
            stack_flags.append(False)
        else:
            raise SystemExit(f"Invalid --stack-rows value: {s!r} (use true/false)")

    easyocr_configs = [
        EasyOCRGridConfig(detect_network=net, canvas_size=cs, stack_rows=st)
        for net, cs, st in product(net_list, args.canvas_sizes, stack_flags)
    ]

    results: list[dict[str, Any]] = []
    ref_w = max(0.0, min(1.0, args.ref_weight))

    def run_config_block(
        *,
        label: str,
        ocr_engine: str,
        apply_before: Any,
        extra: dict[str, Any],
    ) -> None:
        apply_before()
        for _ in range(max(0, args.warmup)):
            _clear_row_cache()
            _run_dataset_once(paths, ocr_engine)
        _clear_row_cache()
        parse_samples: list[float] = []
        total_samples: list[float] = []
        last_per_image: list[dict[str, Any]] | None = None
        for _ in range(max(1, args.repeats)):
            if args.clear_cache_each_pass:
                _clear_row_cache()
            per_image, psum, tsum = _run_dataset_once(paths, ocr_engine)
            parse_samples.append(psum)
            total_samples.append(tsum)
            last_per_image = per_image

        assert last_per_image is not None
        med_parse = float(statistics.median(parse_samples))
        med_total = float(statistics.median(total_samples))
        n_im = len(paths)
        med_parse_per_img = med_parse / max(n_im, 1)
        med_total_per_img = med_total / max(n_im, 1)
        scores = _score_run(last_per_image, reference)
        mean_comp = float(scores["mean_completeness"])
        mean_ref = scores["mean_reference_accuracy"]
        rank = _balanced_rank(med_parse_per_img, mean_comp, mean_ref, ref_weight=ref_w)
        results.append(
            {
                "label": label,
                "ocr_engine": ocr_engine,
                "n_images": n_im,
                "image_basenames": images_csv,
                "median_parse_ms_sum": med_parse,
                "median_total_ms_sum": med_total,
                "median_parse_ms_per_image": med_parse_per_img,
                "median_total_ms_per_image": med_total_per_img,
                "mean_completeness": mean_comp,
                "mean_reference_accuracy": mean_ref,
                "balanced_rank": rank,
                "effective_detect_network": vkt._easyocr_detect_network,
                "last_run_detail": _serialize_last_run(last_per_image),
                **extra,
            }
        )

    for eng in args.engines:
        if eng == "easyocr":
            for cfg in easyocr_configs:
                run_config_block(
                    label=f"easyocr net={cfg.detect_network} canvas={cfg.canvas_size} stack={cfg.stack_rows}",
                    ocr_engine="easyocr",
                    apply_before=lambda c=cfg: _apply_easyocr(c),
                    extra={
                        "detect_network_requested": cfg.detect_network,
                        "canvas_size": cfg.canvas_size,
                        "stack_rows": cfg.stack_rows,
                    },
                )
        elif eng in ("tesseract", "both"):

            def _apply_tesseract_only() -> None:
                vkt._easyocr_invalidate_reader()
                _clear_row_cache()

            run_config_block(
                label=f"{eng} (no EasyOCR grid)",
                ocr_engine=eng,
                apply_before=_apply_tesseract_only,
                extra={
                    "detect_network_requested": None,
                    "canvas_size": None,
                    "stack_rows": None,
                    "degraded_to_craft": None,
                },
            )

    for r in results:
        if r["ocr_engine"] == "easyocr":
            req = r.get("detect_network_requested")
            eff = r.get("effective_detect_network")
            r["degraded_to_craft"] = bool(req == "dbnet18" and eff == "craft")
        else:
            r["degraded_to_craft"] = False
            r["effective_detect_network"] = None

    results.sort(
        key=lambda x: (-x["balanced_rank"], x["median_parse_ms_per_image"])
    )

    payload = {
        "images": [str(p) for p in paths],
        "max_images": args.max_images,
        "cpu_only": bool(args.cpu),
        "networks_requested": raw_networks,
        "networks_used": net_list,
        "dbnet_sweep_note": skip_note,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "clear_cache_each_pass": args.clear_cache_each_pass,
        "reference_file": args.reference,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "rank_score",
            "label",
            "ocr_engine",
            "n_images",
            "image_basenames",
            "detect_network_requested",
            "effective_detect_network",
            "canvas_size",
            "stack_rows",
            "degraded_to_craft",
            "median_parse_ms_per_image",
            "median_total_ms_per_image",
            "median_parse_ms_sum",
            "median_total_ms_sum",
            "mean_completeness",
            "mean_reference_accuracy",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in results:
                row = {k: r.get(k) for k in fields}
                row["rank_score"] = r["balanced_rank"]
                w.writerow(row)

    print(f"Wrote {out_path} ({len(results)} configs). Top by balanced_rank:")
    for r in results[:8]:
        ref_s = r["mean_reference_accuracy"]
        ref_part = f" ref={ref_s:.2f}" if ref_s is not None else ""
        deg = " [fallback craft]" if r.get("degraded_to_craft") else ""
        print(
            f"  {r['balanced_rank']:.2f}  parse/img ~{r['median_parse_ms_per_image']:.1f} ms"
            f"  (sum {r['median_parse_ms_sum']:.0f} ms / {r['n_images']} img)"
            f"  comp={r['mean_completeness']:.2f}{ref_part}  {r['label']}{deg}"
        )


if __name__ == "__main__":
    main()
