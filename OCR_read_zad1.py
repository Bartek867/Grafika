import cv2
import pytesseract
import numpy as np
import torch

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

image_path = r'D:\WSB\Grafika_3sem\obraz.jpg'
image = cv2.imread(image_path)

if image is None:
    print("❌ Nie udało się wczytać obrazu.")
    exit()

original = image.copy()

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)


contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    roi = image[y:y+h, x:x+w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh = cv2.bitwise_not(thresh)

    config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(thresh, config=config)

    print("📷 Odczytany tekst:", text.strip())

    # Podgląd
    cv2.imshow("ROI - Znak", roi)
    cv2.imshow("Szarość + progowanie", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Nie znaleziono znaku")
