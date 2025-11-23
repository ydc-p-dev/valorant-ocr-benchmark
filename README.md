# Valorant Overlay Kill/Death Tracker

## Kill/Death Detection Demo:
https://drive.google.com/file/d/1NUj_Ucyr-1mkolp1d6s4uZM7t7HWL4Ad/view?usp=drive_link

![Detection Preview](Detection.gif)

## 🎯 Overview
This project is a **real‑time Valorant kill/death tracker** that uses:
- **Screen capture (MSS)**
- **Image processing (OpenCV)**
- **OCR text recognition (Tesseract)**

It reads the killfeed directly from the player’s screen, detects your name in kill/death events, and updates a simple **overlay file (overlay_stats.txt)** that streamers or apps like **OBS** can display.

---

## 🧠 Why This Project Exists
The Valorant API does **not** allow real‑time killfeed access during gameplay.  
Most solutions rely on:
- External hardware
- Paid game overlays
- Fragile memory-reading hacks

This project solves the problem by using **computer vision + OCR** to detect green killfeed boxes on the screen in real time.

### ✔ What It Achieves
- Detects **kills** when your name appears on the killfeed
- Detects **deaths** when your name appears on the opposing killfeed
- Provides **accurate live stats** without external tools
- Works **entirely locally**, fast, and lightweight

---

## ⚙️ Components Used
### **1. MSS (Screen Capture)**
Captures small regions of the screen (killfeed only) with minimal performance impact.

### **2. OpenCV**
- Detects green killfeed highlight boxes
- Finds bounding boxes using contours
- Extracts regions of interest to send to OCR

### **3. Tesseract OCR**
- Reads the text inside each green killfeed box
- Cleans and filters the OCR result

### **4. Python Logic**
Ensures:
- Duplicate events are not counted
- Kill/death timing is respected
- Results are stored to `overlay_stats.txt`

---

## 🚀 How It Works (High Level)
1. Capture two screen regions:
   - **Kills region** (where your kills appear)
   - **Deaths region** (where deaths appear)

2. Find green-highlighted killfeed boxes using HSV masking.
3. Crop each detected box.
4. Run OCR to extract player names.
5. If the text matches your in-game name:
   - Increase **kills** or **deaths** counter.
6. Save results to `overlay_stats.txt`.
7. OBS or any overlay tool reads the text file and displays it.

---

## 🛠 Installation
1. Install Python 3.9+
2. Install dependencies:
```bash
pip install opencv-python numpy mss pytesseract
```
3. Install Tesseract OCR:  
Download for Windows: https://github.com/tesseract-ocr/tesseract

4. Edit these values in the script:
```python
PLAYER_NAME = "LoKee"
```
(Optional)
Adjust screen regions if your resolution/layout differs.

---

## ▶️ Running the Program
Simply run:
```bash
python valorant_killfeed_tracker.py
```
You will see preview windows showing the kill region, death region, and their masks.

The script creates or updates:
```
overlay_stats.txt
```
which contains:
```
Kills: X | Deaths: Y
```

---

## 🖥️ Using with OBS
1. Add a **Text (GDI+)** source
2. Choose `overlay_stats.txt`
3. Enable **"Read from file"**

Your kill/death count will now update live in your stream.

---

## 🔧 Tuning Settings
These are the most important variables:
```python
REFRESH_DELAY = 0.7     # How often screen is scanned
EVENT_COOLDOWN = 4.5    # Minimum time before counting another event
```
Increase these if you get **duplicate kills**.
Decrease these if it **misses some kills**.

---

## 🧩 Known Limitations
- Multi-kills within 0.3 seconds can appear identical in OCR
- Rarely misses kills when green highlight is faint
- OCR accuracy depends on resolution/quality

---

## 🗺 Roadmap / Future Additions
- Add **multi‑kill logic** using rolling history frames
- Add **weapon detection** from icons
- Build a **GUI toggle app**
- Compile into a Windows **.exe**

---

## 🤝 Contributing
Feel free to open issues or submit pull requests.

---

## 📄 License
MIT License (free to use and modify).

