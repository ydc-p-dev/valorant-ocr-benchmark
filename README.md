**Project**: Valorant Killfeed OCR Overlay

- **Purpose**: Capture Valorant killfeed entries from two screen regions, detect green kill/death boxes, OCR player names, and write a simple overlay stats file `overlay_stats.txt` with counts.

**Quick Start**
- Install dependencies (recommended in a virtualenv):

```powershell
pip install opencv-python numpy mss pytesseract
```

- Install Tesseract OCR (Windows):
  - Download and install from https://github.com/tesseract-ocr/tesseract
  - If not on PATH, uncomment and update the path near the top of `valorant_killfeed_tracker.py`:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

- Run the tracker locally (not headless):

```powershell
python .\valorant_killfeed_tracker.py
```

**Configuration**
- Edit `valorant_killfeed_tracker.py` top constants:
  - `PLAYER_NAME` — set to your in-game name (case-insensitive).
  - `REGION_KILL` and `REGION_DEATH` — screen coordinates (top/left/width/height) of the killfeed regions for your monitor and HUD.
  - Color thresholds (`GREEN_LOW`, `GREEN_HIGH`) and size filters (`MIN_*`) — tune if detection misses/false positives.
  - `REFRESH_DELAY` — capture interval (seconds). Lower = more responsive, higher CPU.

**Tuning Tips**
- If you miss quick multi-kills, lower `REFRESH_DELAY` and/or increase capture region height.
- If you get duplicates/false counts, increase size thresholds or adjust `GREEN_LOW/HIGH` to better match the green hue in your HUD.
- Tesseract OCR can be sensitive to size/contrast — see `README` section below for robust OCR ideas.

**OCR robustness (recommended)**
- Upscale small crop images before OCR (makes text clearer to Tesseract).
- Apply CLAHE and adaptive thresholding to improve contrast.
- Use `pytesseract.image_to_data()` to read confidence scores and prefer high-confidence words.

**Files**
- `valorant_killfeed_tracker.py` — main script (captures regions, detects green boxes, OCR, prints counts, writes `overlay_stats.txt`).
- `overlay_stats.txt` — simple text file written each loop; external overlays/widgets can read this to display live stats.

**Placeholders for GIFs / Examples**

- Example: running tracker and detection (insert GIF here):

![gif-detection-1](gifs/detection_run.gif)


- Example: tuning regions (insert GIF here):

![gif-tune-regions](gifs/tune_regions.gif)


**Troubleshooting**
- "No module named cv2" — install `opencv-python`.
- Tesseract not found — ensure it is installed and `pytesseract.pytesseract.tesseract_cmd` points to the binary.
- If the script is slow, consider increasing `REFRESH_DELAY` or offloading OCR to a background thread.

**How to contribute / experiment safely**
- Make a backup before large edits: `copy valorant_killfeed_tracker.py valorant_killfeed_tracker.py.bak`
- Add small, reversible changes and test with short sessions.

**License / Attribution**
- Personal project. Add your preferred license if you plan to share.

---

If you want, I can (A) add the README to git, commit and push (I will attempt a push and report any auth/remote issues), or (B) also add a small `requirements.txt` and a `.gitignore`. Which do you prefer?