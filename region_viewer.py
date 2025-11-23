import cv2
import numpy as np
import mss

# --- INITIAL REGION (edit freely) ---
REGION = {"top": 80, "left": 1450, "width": 430, "height": 230}

print("Move or adjust REGION values manually in the script.")
print("Press Q to quit and print the current REGION values.\n")

with mss.mss() as sct:
    while True:
        frame = np.array(sct.grab(REGION))
        cv2.imshow("Region Preview", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nCurrent REGION values:")
            print(REGION)
            break

cv2.destroyAllWindows()
