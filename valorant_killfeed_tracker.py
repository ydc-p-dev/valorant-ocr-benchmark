import cv2
import cv2
import numpy as np
import mss
import time
import pytesseract
import difflib
import re

# If Tesseract isn’t in PATH, set it manually
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- CONFIG ---
REGION_KILL  = {"top": 80, "left": 1375, "width": 400, "height": 300}   # region when YOU kill
REGION_DEATH = {"top": 80, "left": 1775, "width": 150, "height": 300}   # region when YOU die

PLAYER_NAME = "LoKee"   # <-- your in-game name (case-insensitive)

GREEN_LOW = np.array([35, 40, 80])
GREEN_HIGH = np.array([90, 255, 255])

MIN_WIDTH_KILL = 80
MIN_HEIGHT_KILL = 25
MIN_WIDTH_DEATH = 30
MIN_HEIGHT_DEATH = 20
ASPECT_RATIO_MIN = 1.5
ASPECT_RATIO_MAX = 10

MIN_GREEN_PIXELS_KILL = 150
MIN_GREEN_PIXELS_DEATH = 40

REFRESH_DELAY = 0.2
EVENT_COOLDOWN = 5

kills = 0
deaths = 0
last_kill_time = 0
last_death_time = 0

# --- OCR Helper ---
def extract_text_from_region(img):
    """Run OCR on given image region and return cleaned text."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray, config="--psm 6").strip()
    cleanText = clean_text(text)
    return cleanText

def clean_text(text):
    """Remove junk OCR text and keep only words with similar length to the player name."""
    # Remove non-alphabetic chars and normalize spaces
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Split into words and keep only those similar in length to player name
    min_len = max(3, len(PLAYER_NAME) - 1)   # a bit lenient (allow one char shorter)
    filtered_words = [w for w in text.split() if len(w) >= min_len]

    # Join them back into a single clean string
    return " ".join(filtered_words)



def name_match(detected_text, target_name):
    """Check if OCR text contains or closely matches your name."""
    detected_text = detected_text.lower()
    target_name = target_name.lower()
    # Use fuzzy ratio for partial match
    ratio = difflib.SequenceMatcher(None, detected_text, target_name).ratio()
    return target_name in detected_text or ratio > 0.75


# --- Detection ---
def detect_green_boxes(frame, tag="Kill"):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    if tag.lower() == "kill":
        min_w, min_h, min_area = MIN_WIDTH_KILL, MIN_HEIGHT_KILL, MIN_GREEN_PIXELS_KILL
    else:
        min_w, min_h, min_area = MIN_WIDTH_DEATH, MIN_HEIGHT_DEATH, MIN_GREEN_PIXELS_DEATH

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h)
        if w > min_w and h > min_h and ASPECT_RATIO_MIN < aspect < ASPECT_RATIO_MAX:
            boxes.append((x, y, w, h))

    return boxes, mask


# --- Main Loop ---
with mss.mss() as sct:
    print("🎯 Dual-Region Killfeed + OCR Detection running... Press Q to stop.")
    time.sleep(1)

    while True:
        frame_kill = np.ascontiguousarray(np.array(sct.grab(REGION_KILL))[:, :, :3])
        frame_death = np.ascontiguousarray(np.array(sct.grab(REGION_DEATH))[:, :, :3])

        boxes_kill, mask_kill = detect_green_boxes(frame_kill, tag="Kill")
        boxes_death, mask_death = detect_green_boxes(frame_death, tag="Death")

        now = time.time()

        # --- OCR check for Kill region ---
        for (x, y, w, h) in boxes_kill:
            cropped = frame_kill[y:y+h, x:x+w]
            detected_text = extract_text_from_region(cropped)
            if name_match(detected_text, PLAYER_NAME) and now - last_kill_time > EVENT_COOLDOWN:
                kills += 1
                last_kill_time = now
                print(f"🟩 Kill confirmed! ({detected_text}) | Total: {kills}")
            cv2.rectangle(frame_kill, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # --- OCR check for Death region ---
        for (x, y, w, h) in boxes_death:
            cropped = frame_death[y:y+h, x:x+w]
            detected_text = extract_text_from_region(cropped)
            if name_match(detected_text, PLAYER_NAME) and now - last_death_time > EVENT_COOLDOWN:
                deaths += 1
                last_death_time = now
                print(f"🟥 Death confirmed! ({detected_text}) | Total: {deaths}")
            cv2.rectangle(frame_death, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # --- Show preview ---
        cv2.imshow("Kill Region", frame_kill)
        cv2.imshow("Death Region", frame_death)
        cv2.imshow("Kill Mask", mask_kill)
        cv2.imshow("Death Mask", mask_death)

        # --- Save overlay ---
        with open("overlay_stats.txt", "w") as f:
            f.write(f"Kills: {kills} | Deaths: {deaths}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(REFRESH_DELAY)

    cv2.destroyAllWindows()