import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from valorant_killfeed_tracker import (
    REGION_KILLFEED,
    detect_killfeed_row_boxes,
    load_bgr,
    process_frame,
)


def crop_killfeed_region(full_img: np.ndarray) -> np.ndarray:
    h, w = full_img.shape[:2]
    x = max(0, min(REGION_KILLFEED["left"], w - 1))
    y = max(0, min(REGION_KILLFEED["top"], h - 1))
    rw = max(1, min(REGION_KILLFEED["width"], w - x))
    rh = max(1, min(REGION_KILLFEED["height"], h - y))
    return full_img[y : y + rh, x : x + rw]


def ahash_u64(img: np.ndarray) -> str:
    """Tiny perceptual hash for fast near-duplicate clustering."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    mean = float(small.mean())
    bits = (small > mean).flatten().astype(np.uint8)
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:016x}"


def hamming_hex64(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def list_images(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline killfeed indexer: fast candidate filtering + optional OCR parsing."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="killfeed_screenshots",
        help="Folder with full-screen screenshots.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="offline_killfeed_index.jsonl",
        help="Per-image index (JSON Lines).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="offline_killfeed_index.csv",
        help="Per-image index (CSV).",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Only do fast candidate detection, no OCR parsing.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit number of images (0 = all).",
    )
    parser.add_argument(
        "--dedupe-threshold",
        type=int,
        default=3,
        help="Hamming distance threshold for near-duplicate ROI hash (0-64).",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.exists():
        raise SystemExit(f"Input directory not found: {in_dir}")

    paths = list_images(in_dir)
    if args.max_files > 0:
        paths = paths[: args.max_files]
    if not paths:
        raise SystemExit("No images found.")

    out_jsonl = Path(args.output_jsonl)
    out_csv = Path(args.output_csv)

    total = len(paths)
    kept_candidates = 0
    skipped_dupe = 0
    records: list[dict] = []

    prev_hash = None
    t_all_0 = time.perf_counter()
    with open(out_jsonl, "w", encoding="utf-8") as jf:
        for idx, p in enumerate(paths, start=1):
            t0 = time.perf_counter()
            img = load_bgr(p)
            roi = crop_killfeed_region(img)
            roi_hash = ahash_u64(roi)

            near_duplicate = False
            if prev_hash is not None:
                near_duplicate = hamming_hex64(roi_hash, prev_hash) <= args.dedupe_threshold
            prev_hash = roi_hash

            boxes, masks = detect_killfeed_row_boxes(roi)
            green_pixels = int(np.count_nonzero(masks["green"]))
            red_pixels = int(np.count_nonzero(masks["red"]))
            has_candidate = len(boxes) > 0

            events_count = 0
            you_kills = 0
            you_deaths = 0
            sample_event = ""
            ocr_total_ms = 0.0

            if has_candidate and not near_duplicate:
                kept_candidates += 1
                if not args.skip_ocr:
                    events, _, timing, _masks = process_frame(
                        roi,
                        now=time.time(),
                        recent_pairs=None,
                        draw=None,
                    )
                    events_count = len(events)
                    ocr_total_ms = float(timing.get("t_parse_ms_total", 0.0))
                    if events:
                        sample_event = f"{events[0].killer} -> {events[0].victim}"
            elif near_duplicate:
                skipped_dupe += 1

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            rec = {
                "file": p.name,
                "has_candidate": has_candidate,
                "near_duplicate": near_duplicate,
                "candidate_rows": len(boxes),
                "green_pixels": green_pixels,
                "red_pixels": red_pixels,
                "green_rows": sum(1 for b in boxes if b[4] == "green"),
                "red_rows": sum(1 for b in boxes if b[4] == "red"),
                "events_count": events_count,
                "roi_hash": roi_hash,
                "sample_event": sample_event,
                "analysis_ms": round(elapsed_ms, 3),
                "ocr_ms": round(ocr_total_ms, 3),
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.append(rec)

            if idx % 25 == 0 or idx == total:
                print(f"[{idx}/{total}] processed")

    write_csv(out_csv, records)
    total_ms = (time.perf_counter() - t_all_0) * 1000.0

    print("\nOffline indexing complete")
    print(f"Input images: {total}")
    print(f"Candidates kept: {kept_candidates}")
    print(f"Near-duplicates skipped for OCR: {skipped_dupe}")
    print(f"JSONL: {out_jsonl}")
    print(f"CSV:   {out_csv}")
    print(f"Total time: {total_ms / 1000.0:.2f}s")


if __name__ == "__main__":
    main()

