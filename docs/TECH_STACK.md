# Technologies and approaches (record)

This file is the **project’s consolidated record** of stack and design choices. Detailed narratives live in the linked docs.

---

## Runtime and packaging

| Item | Notes |
|------|--------|
| **Python 3.9+** | CLI scripts, library-style modules |
| **pip** | Dependencies per README (`opencv-python`, `numpy`, `mss`, `pytesseract`, optional `easyocr` / `torch`) |

---

## Screen input and outputs

| Item | Role |
|------|------|
| **mss** | Low-overhead grab of a monitor rectangle (`REGION_KILLFEED`) in live mode |
| **File / bytes / ndarray** | Static and API-style input for [`parse_killfeed.py`](../parse_killfeed.py) (`parse_killfeed_image`) |
| **JSON** | `killfeed_events.json`, `killfeed_parse.json`, benchmark and reference files |
| **JSONL** | `analysis_timings.jsonl` (per-frame / per-image timings) |
| **Plain text** | `overlay_stats.txt` for OBS / external tools |
| **PNG** | Optional full-screen or debug dumps (`temp/screenshots/`, `assets/screenshots/` benchmarks, `--debug-out`) |

---

## Computer vision (OpenCV + NumPy)

| Approach | Where used |
|----------|------------|
| **BGR → HSV** | Green / red killfeed highlight masks; white-text heuristics for OCR split and Tesseract preprocess |
| **Morphology** | Noise cleanup on masks |
| **Contours + bounding boxes** | **Live tracker** row detection when not using fixed bands |
| **Fixed horizontal bands** | `row_bands_frac` from `config/killfeed_row_bands.json` — unified parser and weapon matcher share the same geometry |
| **ROI crop + expand_row_box** | Per-row strips fed to OCR |
| **Template matching** | Weapon icons: **masked normalized cross-correlation** (`TM_CCORR_NORMED`), alpha-aware — see [`WEAPON_MATCHING.md`](WEAPON_MATCHING.md) |

---

## OCR

| Engine | Role |
|--------|------|
| **Tesseract** (via **pytesseract**) | CPU; whitelist + PSM; multi-preprocess (Otsu, adaptive, HSV “white” mask), best variant scored |
| **EasyOCR** | Default engine in repo; **CRAFT** detector (default); optional **DBNet**; GPU via **PyTorch + CUDA** |
| **Hybrid `both`** | Fuses Tesseract and EasyOCR per side with a simple score heuristic |

**Approaches:** nickname-oriented character allowlist (incl. space for multi-word names in EasyOCR); `normalize_ocr_text`; LRU **row OCR cache** for repeated identical crops; **stacked multi-row** `readtext` in live `process_frame` (not used the same way in unified per-row parse). Timing and grid tuning: [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md).

---

## Weapon classification (optional)

| Item | Notes |
|------|--------|
| **Config** | `config/weapon_templates.json` → PNG paths under `assets/icons/` |
| **Algorithm** | Multi-scale template match, NMS peaks, margin vs runner-up, optional “white weapon slot” crop — [`WEAPON_MATCHING.md`](WEAPON_MATCHING.md) |
| **Integration** | `scripts/match_killfeed_weapon.py`; same bands inside [`parse_killfeed_roi_unified`](../parse_killfeed.py) |

---

## Unified parse (names + weapons)

| Item | Notes |
|------|--------|
| **Entry** | `parse_killfeed_image`, `parse_killfeed_roi_unified`, CLI `parse_killfeed.py` |
| **Pipeline** | Fixed bands → HSV activity gate → weapon pass (skipped if inactive) → OCR on active rows → `KillfeedEvent` |
| **Doc** | [`UNIFIED_PARSE.md`](UNIFIED_PARSE.md) |

---

## Live tracker-specific logic

| Approach | Purpose |
|----------|---------|
| **Deduplication** | Drop repeated killer→victim within a short time window |
| **Tall-row split** | Split one HSV contour that spans two killfeed lines |
| **Fragment row prune** | Remove spurious strips where killer duplicates another row’s victim with empty victim |
| **Warm-up** | `warm_easyocr_for_session` before first real frame |

---

## Tooling and benchmarks

| Script | Purpose |
|--------|---------|
| `benchmark_killfeed_ocr.py` | EasyOCR grid + optional reference accuracy |
| `benchmark_parse_killfeed.py` | End-to-end unified parse timings |
| `offline_killfeed_indexer.py` | Batch folder indexing |

---

## Related documentation

| Doc | Focus |
|-----|--------|
| [`README.md`](../README.md) | Install, GPU, quick start |
| [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md) | OCR benchmark methodology and §7 tech table |
| [`WEAPON_MATCHING.md`](WEAPON_MATCHING.md) | Weapon NCC pipeline and tuning |
| [`UNIFIED_PARSE.md`](UNIFIED_PARSE.md) | Unified parser API and pipeline |

---

*Update this file when you add a new major dependency (e.g. database layer) or change the default OCR / detection stack.*
