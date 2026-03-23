# Killfeed weapon template matching

This document describes how `scripts/match_killfeed_weapon.py` classifies weapon icons in the killfeed ROI using **masked normalized cross-correlation** (OpenCV `TM_CCORR_NORMED` with a template alpha mask). It also records **parameter choices that worked well** on sample 1080p-style captures in this repo.

## Pipeline (multi-row)

1. **ROI** — Default killfeed rectangle matches `REGION_KILLFEED` in `valorant_killfeed_tracker.py` (full-monitor coordinates). Override with `--killfeed-rect TOP,LEFT,WIDTH,HEIGHT` or `--no-crop` for a pre-cropped strip.

2. **Row strips** — With `--row-bands-json`, each row is a horizontal band as fractions of ROI height (`row_bands_frac`). This avoids matching across stacked kill lines at once.

3. **Center crop** — `--center-frac` (e.g. `0.35`) keeps the middle portion of each row strip horizontally so search stays in the weapon column and away from far edges.

4. **White weapon slot** — With multiple strips, `--weapon-white-slot` is on by default: bright pixels are thresholded, a gun-shaped connected component is chosen between name regions, and matching runs in that **horizontal slice** first.

5. **Slot width vs templates** — Rifle templates are wider than SMG-sized ones. If the detected white blob slice is **narrower than the smallest scaled template width** for any weapon in `--templates-json`, `matchTemplate` cannot run for those weapons inside the slice only. The script **widens the slice symmetrically** to `min_slot_dims_for_all_templates` (per weapon: minimum template width/height across your `--scales`, then max across weapons). That uses the **smallest** scale per weapon so the crop does not balloon to the largest scale (which would pull in extra context and collapse Phantom vs Vandal margins).

6. **Peaks and classification** — For each weapon and scale, NMS peaks above `--min-score` are collected. **Winner** = best score among weapons that have a real peak in the voting region (`max_scores_by_weapon_right_of_strip`, or near-anchor fallback). **Runner-up** for margin uses scores after optional **supplement**:

7. **Supplement class scores** — On by default (`--supplement-class-scores`). For templates missing from peak votes, the script adds the **global max NCC** over a gray image. That image is the **full row center strip**, not only the narrow slot, so missing rifles still get a score for margin checks (fixes “only Spectre in slot” margin collapse). **Important:** supplement must **not** pick the label; only peaks inside the match window do. That avoids labeling unknown pistols as the SMG when supplement alone would dominate.

8. **Margin gate** — With two or more templates, require `best_peak_score − runner_up_score ≥ --min-class-margin` (unless margin is off). **Tie-break slack:** if the winner’s NCC is **≥ 0.982** and the margin misses the threshold by at most **0.00085**, the row is still accepted. This covers **Vandal vs Phantom** on the same white silhouette where NCC differs by ~0.001.

9. **Geometry / anchor radius** — Classification picks a bounding box via `best_geometry_for_weapon` around the weapon anchor. For white-slot rows, the match crop can be wider than the old `min(classify_r, 0.42·slot_width)` radius; the script uses a radius derived from the **half-diagonal** of the match rectangle (plus a floor) so the winning peak is still inside the search disk.

## Recommended CLI (validated on project screenshots)

These settings were exercised on captures such as multi-row Vandal/Phantom/Spectre rows, a narrow-slot Vandal frame, and a pistol-only row (unknown weapon — should **not** label as a known rifle/SMG).

```bash
python scripts/match_killfeed_weapon.py path/to/screenshot.png \
  --templates-json config/weapon_templates.json \
  --row-bands-json config/killfeed_row_bands.json \
  --center-frac 0.35 \
  --scales 0.25,0.35,0.45 \
  --min-score 0.25 \
  --min-class-margin 0.011 \
  --draw assets/debug_weapon_match.png
```

| Parameter | Suggested value | Role |
|-----------|-----------------|------|
| `--center-frac` | `0.35` | Focuses on the central weapon column of each row strip. |
| `--scales` | `0.25,0.35,0.45` | Covers icon size variation; smallest scale must fit your templates in the (possibly widened) slot. |
| `--min-score` | `0.25` | TM_CCORR_NORMED floor for NMS peaks (lower than script default `0.38` for small / soft icons). |
| `--min-class-margin` | `0.010`–`0.011` | Rejects ambiguous rows; `0.011` plus built-in slack handles tight Vandal/Phantom ties. |
| `--weapon-white-slot` | default **on** | Restricts search to the bright gun blob when multiple row bands are used. |
| `--supplement-class-scores` | default **on** | Keeps runner-up scores meaningful; use `--no-supplement-class-scores` if a specific frame misbehaves. |

**Config files:**

- `config/killfeed_row_bands.json` — `row_bands_frac` aligned to ROI height (see `_comment` in file).
- `config/weapon_templates.json` — map weapon id → PNG path (silhouette / alpha mask; see `process_weapon_icon.py`). Shipped templates live under **`assets/icons/`**.

## Tuning guide

- **False weapon on unknown guns** — Raise `--min-class-margin`, tighten `--center-frac`, or disable supplement for that workflow. Winner-from-peaks + supplement on full strip is designed to reject many “wrong SMG” cases on pistols.

- **Missed real weapon** — Slightly lower `--min-class-margin` (e.g. `0.010`) or improve template quality / alignment. Check `--draw` output: magenta box = white slot (search band); green box = accepted hit.

- **Row without white slot** — Script falls back to full center strip for that row (`classify-vote-min-x-frac` applies). Warnings like `white weapon slot not found` are expected on some rows.

- **Phantom vs Vandal always tied** — Use crisper templates, or rely on the small high-confidence slack; do not disable margin entirely if you care about false positives.

## Exit codes

- `0` — At least one weapon hit passed all gates.
- `2` — No qualifying hits (useful for scripting).

## Unified parser (`parse_killfeed.py`)

Full pipeline (library API, `KillfeedEvent`, inactive bands, prototype usage): **[`docs/UNIFIED_PARSE.md`](UNIFIED_PARSE.md)**.

Single entry point that runs **OCR + weapon templates** on the same row bands:

```bash
python parse_killfeed.py --image path/to/screenshot.png \
  --row-bands-json config/killfeed_row_bands.json \
  --weapon-templates-json config/weapon_templates.json \
  --debug-out assets/debug_killfeed/unified.png
```

Outputs **`killfeed_parse.json`** (array of rows with `killer`, `victim`, `weapon`, `weapon_score`, `row_band_index`, `active`, …). Rows with no killfeed highlight in that band (empty slot over the map) get **`active`: false**, empty names, `row_color`: `"inactive"`, and skip OCR/weapons — tune **`--min-band-highlight-px`** (default 280) and **`--min-band-highlight-frac`** if needed. **`--omit-inactive-bands`** writes only active rows.

Use **`--no-weapons`** for names only. Defaults assume repo `config/` paths.

## Shared config with killer / victim OCR

`valorant_killfeed_tracker.py` supports the same file:

```bash
python valorant_killfeed_tracker.py --image path/to/screenshot.png \
  --row-bands-json config/killfeed_row_bands.json --no-show
```

Add **`--debug-dir path/to/folder`** to save PNGs: `*_annotated.png` (gray rectangles = all detected row bands, colored = expanded OCR boxes + labels), `*_mask_green.png`, `*_mask_red.png`, `*_masks_sidebyside.png`. In **live** mode, the same set is written only when the frame produces at least one parsed event.

Each `row_bands_frac` entry becomes a **full-width** row crop for OCR. Row color (`green` / `red`) is chosen by comparing green vs red HSV mask pixel counts inside that band (no contour-based row detection). Timing logs include `row_detect_mode`: `"fixed_bands"` vs `"hsv_contours"`.

## See also

- **[`docs/TECH_STACK.md`](TECH_STACK.md)** — consolidated technologies and approaches for the whole repo.
- Module docstring at top of `scripts/match_killfeed_weapon.py` for extra examples and flags.
- `docs/BENCHMARK_REPORT.md` if you add timing notes for `--bench-repetitions`.
