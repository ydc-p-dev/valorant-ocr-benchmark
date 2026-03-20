# Killfeed OCR benchmark and recommended settings

Document date: March 2026. Sources: [bench_kf3.csv](../bench_kf3.csv), [benchmark_killfeed_results_kf3.json](../benchmark_killfeed_results_kf3.json), [bench_kf6.csv](../bench_kf6.csv), [benchmark_killfeed_results_kf6.json](../benchmark_killfeed_results_kf6.json), [benchmark_reference_kf6.json](../benchmark_reference_kf6.json), [bench_kf6_engines.csv](../bench_kf6_engines.csv), [benchmark_single_110639.json](../benchmark_single_110639.json), settings in [valorant_killfeed_tracker.py](../valorant_killfeed_tracker.py).

---

## 1. Purpose of the benchmark

[benchmark_killfeed_ocr.py](../benchmark_killfeed_ocr.py) sweeps EasyOCR parameter combinations on a set of static screenshots, measures parse time and (optionally) accuracy against a reference JSON. That lets you compare **speed** and **quality** without manually restarting with different constants in code.

---

## 2. Screenshot resolution

All images used for the numbers in this report (kf3, kf6, plus the single-image `110639` run) were checked on disk:

- **Full-frame PNG:** **1920×1080 px** (Full HD).

The tracker then crops the killfeed column using a code constant (defaults):

- **`REGION_KILLFEED`:** width **600 px**, height **180 px** (short strip — avoids the **combat report** modal below the killfeed), `left` / `top` for your monitor (typically `left: 1300`, `top: 80` — see [valorant_killfeed_tracker.py](../valorant_killfeed_tracker.py)). Override per run: `--killfeed-rect TOP,LEFT,WIDTH,HEIGHT`.

OCR runs on **individual row crops** inside that region; their size depends on contour detection.

**Important:** different monitor resolution, Windows display scaling, or region changes will change benchmark numbers and OCR quality — **re-run the benchmark** on representative PNGs from **your** setup.

---

## 3. How time is measured and what “warmup” means

### In the benchmark

- **`warmup`:** before timed passes, one full run over the whole image set. **EasyOCR Reader** loads here; models move into RAM/GPU; part of CUDA init happens here.
- **`repeats`:** several **timed** passes follow; JSON/CSV report the **median** time (robust to outliers).
- **`--clear-cache-each-pass` (default on):** before each timed pass the **row-geometry LRU OCR cache** is cleared so numbers reflect full OCR work, not a cache hit on the same crop.

So the published **tens of ms per image** are typically a **hot** state after warmup in **one** Python process — a **median** over several repeats.

### In the tracker ([valorant_killfeed_tracker.py](../valorant_killfeed_tracker.py))

- Before the first frame (static and live), **`warm_easyocr_for_session()`** runs: synthetic text, **single-row** and **stacked** (two rows) paths matching real killfeed, plus **`torch.cuda.synchronize()`** so GPU warmup finishes.
- **Model load and that warmup** are **not** included in **`t_parse_ms_total`** in [analysis_timings.jsonl](../analysis_timings.jsonl) — only the parse phase after row detection.
- Each `python ... --image one.png` is a **new process**: even after warmup the first **real** frame can still be slower than the benchmark median (tensor shapes, driver work). To compare with the benchmark, run **several files in one process** (`--folder`) and look at timings for the **second and later** lines in [analysis_timings.jsonl](../analysis_timings.jsonl).

---

## 4. Benchmark results (kf3 set)

**Dataset:** 3 images ([killfeed_20260320_110428_934.png](../killfeed_screenshots/killfeed_20260320_110428_934.png), [killfeed_20260320_110237_631.png](../killfeed_screenshots/killfeed_20260320_110237_631.png), [killfeed_20260319_193915_057.png](../killfeed_screenshots/killfeed_20260319_193915_057.png)); details in the `image_basenames` column of [bench_kf3.csv](../bench_kf3.csv).  
**Engine:** EasyOCR, **CRAFT** detector (`dbnet18` was not used on test Windows without a full toolchain — see [README](../README.md)).  
**Grid:** `canvas_size` ∈ {480, 640, 960}, `stack_rows` ∈ {true, false}.

### Top by `balanced_rank` (speed + completeness; reference here 2/3 — see below)

| Rank | Configuration | Median parse, ms/image | Median total, ms/image | mean_completeness | mean_reference_accuracy |
|------|----------------|------------------------|------------------------|-------------------|-------------------------|
| 1 | craft, canvas **480**, **stack=True** | **~39.5** | **~42.3** | 1.0 | 0.667 |
| 2 | craft, canvas 640, stack=True | ~39.7 | ~42.8 | 1.0 | 0.667 |
| 3 | craft, canvas 960, stack=True | ~43.5 | ~46.4 | 1.0 | 0.667 |
| … | stack=False (all canvas sizes) | ~53–56 ms/image | higher | 1.0 | 0.667 |

**Speed takeaway:** best in this grid is **CRAFT + canvas 480 + stacked readtext** for multiple rows. Disabled stack is noticeably slower.

**Quality vs reference (0.667):** all grid configs produced the same OCR pairs; the metric is not 1.0 partly because one frame has **two** killfeed rows while the reference lists **one**, plus nickname spelling differences (for strict checks, align the reference with the real row count).

### kf6 extended set (six images, CUDA)

**Dataset:** kf3 trio plus [killfeed_20260319_193249_052.png](../killfeed_screenshots/killfeed_20260319_193249_052.png), [killfeed_20260319_193548_483.png](../killfeed_screenshots/killfeed_20260319_193548_483.png), [killfeed_20260319_193712_526.png](../killfeed_screenshots/killfeed_20260319_193712_526.png) (multi-row, spaced nickname, empty ROI). Reference: [benchmark_reference_kf6.json](../benchmark_reference_kf6.json). **Host (example):** PyTorch **2.10.0+cu126**, **NVIDIA GeForce GTX 1660 Ti**, `torch.cuda.is_available() == True`. **Sweep:** CRAFT, `canvas_size` ∈ {480, 640, 960}, `stack_rows` ∈ {true, false}, `--repeats 5 --warmup 1`. Artifacts: [bench_kf6.csv](../bench_kf6.csv), [benchmark_killfeed_results_kf6.json](../benchmark_killfeed_results_kf6.json).

**Grid leader (by `balanced_rank`):** **craft, canvas 640, stack=True** — **~29.6 ms/image** parse (~31.6 ms total/image). On this run **stack=True** occupied the top three slots; **stack=False** was ~41–42 ms/image (clearly slower on GPU for this set). **mean_reference_accuracy** was **0.40** and **mean_completeness ~0.94** for every cell (unchanged vs CPU-only kf6 runs — canvas/stack does not fix strict reference mismatches).

| Rank | Config | Median parse, ms/image | Median total, ms/image | mean_completeness | mean_reference_accuracy |
|------|--------|------------------------|------------------------|-------------------|-------------------------|
| 1 | canvas **640**, **stack=True** | **~29.6** | **~31.6** | ~0.94 | 0.40 |
| 2 | canvas 960, stack=True | ~29.9 | ~32.0 | ~0.94 | 0.40 |
| 3 | canvas 480, stack=True | ~32.3 | ~34.7 | ~0.94 | 0.40 |
| 4 | canvas 480, stack=False | ~41.0 | ~43.2 | ~0.94 | 0.40 |
| 5 | canvas 640, stack=False | ~41.1 | ~43.1 | ~0.94 | 0.40 |
| 6 | canvas 960, stack=False | ~42.1 | ~44.2 | ~0.94 | 0.40 |

**Engine spot-check (same six images, EasyOCR fixed to canvas 480 + stack=True):** [bench_kf6_engines.csv](../bench_kf6_engines.csv). EasyOCR **~29.0 ms/image** parse; **both** ~**1039** ms/image; **tesseract** ~**922** ms/image; ref **0.40 / 0.20 / 0.00** respectively.

**Parser post-step (after the CUDA table above):** [valorant_killfeed_tracker.py](../valorant_killfeed_tracker.py) now (1) **splits tall HSV row contours** that often merge two killfeed lines, and (2) **drops fragment rows** where the killer repeats another row’s victim while the victim is empty/`?` (spurious second strip). Re-run the kf6 benchmark to refresh CSV/JSON; on a quick check **`mean_reference_accuracy` moved ~0.40 → ~0.50** on the same reference while **speed tier stays ~30 ms/image** on GPU-class hardware. Remaining gaps are mostly **strict string** mismatches (`hookill` vs `hookill4`, `Bot` vs `Bot 1`, spaces in nicknames) and **extra rows** still detected on some frames (e.g. second red bar on `110428`).

```bash
python benchmark_killfeed_ocr.py --images killfeed_screenshots/killfeed_20260320_110428_934.png killfeed_screenshots/killfeed_20260320_110237_631.png killfeed_screenshots/killfeed_20260319_193915_057.png killfeed_screenshots/killfeed_20260319_193249_052.png killfeed_screenshots/killfeed_20260319_193548_483.png killfeed_screenshots/killfeed_20260319_193712_526.png --reference benchmark_reference_kf6.json --networks craft --canvas-sizes 480 640 960 --stack-rows true false --engines easyocr --repeats 5 --warmup 1 --out benchmark_killfeed_results_kf6.json --csv bench_kf6.csv
```

### Extra check (one image, 10 repeats)

File [benchmark_single_110639.json](../benchmark_single_110639.json): image [killfeed_20260320_110639_612.png](../killfeed_screenshots/killfeed_20260320_110639_612.png), **craft, canvas 480, stack=True**, warmup 1, repeats 10, cache cleared each pass.

- Median **parse** per image: **~33.3 ms**
- Median **total** per image: **~35.9 ms**

That matches the order of magnitude of the kf3 top (one row per frame — a bit less work than averaging over three different frames).

### Engine comparison: EasyOCR vs Tesseract vs Both

Same **3 images** kf3 and reference [**benchmark_reference_kf3.json**](../benchmark_reference_kf3.json), `--repeats 5 --warmup 1`, artifacts: [benchmark_kf3_engines.json](../benchmark_kf3_engines.json), [bench_kf3_engines.csv](../bench_kf3_engines.csv).

Repeat command:

```bash
python benchmark_killfeed_ocr.py --images killfeed_screenshots/killfeed_20260320_110428_934.png killfeed_screenshots/killfeed_20260320_110237_631.png killfeed_screenshots/killfeed_20260319_193915_057.png --reference benchmark_reference_kf3.json --engines easyocr tesseract both --repeats 5 --warmup 1 --out benchmark_kf3_engines.json --csv bench_kf3_engines.csv
```

Summary (median **parse** aggregated over 3 frames / **per image**; `mean_reference_accuracy` vs [benchmark_reference_kf3.json](../benchmark_reference_kf3.json)):

| Engine | Note | parse ≈ ms/image | parse sum 3 img | mean_completeness | mean_reference_accuracy |
|--------|------|------------------|-----------------|-------------------|-------------------------|
| **easyocr** | top: craft, canvas 480, stack=True | **~33** | **~100** | 1.0 | **0.67** |
| easyocr | other grid cells | ~38–54 | ~113–161 | 1.0 | 0.67 |
| **both** | EasyOCR + Tesseract on names | **~1420** | **~4261** | 1.0 | **0.33** |
| **tesseract** | Tesseract only | **~1267** | **~3802** | 1.0 | **0.00** |

**Takeaway:** on this set **EasyOCR (GPU)** wins on speed (order **~30–50 ms/image** after warmup) and on reference match (**2/3** pairs under benchmark rules) over **Tesseract** (~**40×** slower, **0/3** on reference) and **both** (**1/3** on reference). For a GPU live overlay, **`--ocr-engine easyocr`** is the sensible default; Tesseract helps as a fallback without PyTorch/CUDA or for A/B tests.

### EasyOCR: GPU vs CPU (no CUDA)

Same 3 PNGs, kf3 reference, narrow grid **craft + canvas 480 + stack=True**, `--repeats 3 --warmup 1`, engines **easyocr, tesseract, both**. Files: [benchmark_kf3_cpu.json](../benchmark_kf3_cpu.json), [bench_kf3_cpu.csv](../bench_kf3_cpu.csv).

Enable CPU in the benchmark:

```bash
python benchmark_killfeed_ocr.py --images …(three kf3)… --reference benchmark_reference_kf3.json --engines easyocr tesseract both --networks craft --canvas-sizes 480 --stack-rows true --repeats 3 --warmup 1 --cpu --out benchmark_kf3_cpu.json --csv bench_kf3_cpu.csv
```

| Mode | EasyOCR: parse ≈ ms/image | Note |
|------|---------------------------|------|
| **GPU (CUDA)** | **~33–38** | as in “Engine comparison” |
| **CPU (`--cpu`)** | **~260–270** (median across runs) | about **~7–8×** slower than GPU on this set |

Tesseract / **both** on the CPU run stay in the same ballpark (**~1.3–1.5 s/image**): those paths are already CPU-heavy; **reference** scores unchanged (0.00 / 0.33).

To force CPU-only EasyOCR, use **`--cpu`** on the benchmark or set **`CUDA_VISIBLE_DEVICES=-1`** in the environment before `python` (syntax depends on your shell).

---

## 5. Recommended parse settings (grid winner)

| Parameter | Recommended | CLI |
|-----------|-------------|-----|
| OCR engine | **EasyOCR** (GPU if CUDA PyTorch) | `--ocr-engine easyocr` |
| EasyOCR detector | **CRAFT** | `--easyocr-detect-network craft` (repo default) |
| Detection canvas size | **480** | `--easyocr-canvas-size 480` |
| Stacked OCR for multiple rows | **on** | omit `--easyocr-no-stack-rows` |

**kf6 on GTX 1660 Ti (CUDA):** full grid winner was **canvas 640** + stack (see **kf6** subsection above); **480** + stack was ~2.7 ms/image slower on parse — still inside a comfortable 200–500 ms/frame budget. Default **480** stays the main recommendation (matches kf3 grid, slightly smaller detector input); try **`--easyocr-canvas-size 640`** if you want the kf6-tuned optimum on similar hardware.

Single-file example:

```bash
python valorant_killfeed_tracker.py --image killfeed_screenshots/YOUR.png --no-show --ocr-engine easyocr --easyocr-canvas-size 480 --easyocr-detect-network craft
```

**DBNet (`dbnet18`):** on Windows without **CUDA_HOME** and **MSVC (`cl`)**, deformable-conv extensions usually do not build; this repo defaults to **CRAFT**. For dbnet18, install the full toolchain and use `--easyocr-detect-network dbnet18` (see [README](../README.md)).

---

## 6. Nickname casing and line accuracy

OCR sometimes returns a nickname in **different casing** or with small character differences vs the client (e.g. mixed case in Riot style). The benchmark often **lowercases** for reference comparison, so some pairs “pass”, but strict logging or overlay output can still disagree.

**Possible improvement (not implemented by default):** at **match start**, during **loading**, or in **agent select / preload**, when the **roster and player names** are stable on screen, capture that UI region once (or briefly on an interval) and build an **expected nickname list** for the match. Then **match** killfeed OCR strings to that list (case-insensitive exact match, edit distance, constrained allowlist, etc.) and emit **canonical** spellings. That reduces casing noise and minor OCR drift when there is no other ground truth (Valorant does not expose live killfeed to third-party tools).

Limitations: you need a **stable UI region** for lobby/load, handle player changes (reconnect, spectator), and avoid confusing names with other on-screen text.

---

## 7. Technologies used

| Component | Role |
|-----------|------|
| **Python 3.9+** | Logic, CLI, logs |
| **mss** | Screen capture (live) |
| **OpenCV (cv2)** | Region crop, green/red HSV masks, contours, row crops, (optional) preview |
| **NumPy** | Image arrays |
| **pytesseract** + **Tesseract OCR** | Alternative / extra OCR (`--ocr-engine tesseract` or `both`) |
| **EasyOCR** | Fast GPU path: **CRAFT** detector (default), line recognition |
| **PyTorch** | EasyOCR backend; **CUDA** on NVIDIA |
| **JSON / JSONL** | [killfeed_events.json](../killfeed_events.json), [analysis_timings.jsonl](../analysis_timings.jsonl), benchmark outputs |

Also: killer→victim deduplication, [overlay_stats.txt](../overlay_stats.txt) for OBS, optional full-screen PNGs in live mode.

---

## 8. Short summary

1. Best speed balance in the grids we ran: **EasyOCR + CRAFT + canvas 480 + stacked rows**.  
2. **Benchmark time** is a **median** after **warmup** with **row-cache cleared** each pass — closest to “steady” OCR in a long session.  
3. **[analysis_timings.jsonl](../analysis_timings.jsonl)** reflects one frame after **in-process** warmup; a one-off cold process can still differ from the benchmark median.  
4. Stack: **Python, MSS, OpenCV, NumPy, Tesseract (optional), EasyOCR, PyTorch/CUDA**.  
5. On kf3 **Tesseract** and **both** are much slower and worse vs the reference — see the “Engine comparison” table.  
6. **EasyOCR without CUDA** ([benchmark_killfeed_ocr.py](../benchmark_killfeed_ocr.py) `--cpu`): same top config lands around **~260+ ms/image** instead of **~30–40 ms** on GPU — see [bench_kf3_cpu.csv](../bench_kf3_cpu.csv).  
7. **kf6 + CUDA:** grid **leader** = **CRAFT + canvas 640 + stack=True** (~**30 ms/image** parse on GTX 1660 Ti); **stack=True** beats **stack=False** on GPU for this set; reference accuracy stays **0.40** across the grid.  
8. **Post-parse heuristics** (tall-row split + duplicate-strip prune) improve kf6 **reference** score in testing — **re-run** the benchmark after pulling; see kf6 subsection above.

---

*When you change the screenshot set, **resolution**, killfeed region, or reference, re-run [benchmark_killfeed_ocr.py](../benchmark_killfeed_ocr.py) and refresh the tables in this file as needed.*
