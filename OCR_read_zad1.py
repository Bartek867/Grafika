import cv2
import pytesseract
import numpy as np
import torch 
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

pytesseract.pytesseract.tesseract_cmd = r'sciezka' 

image_path = r'sciezka'
image = cv2.imread(image_path)

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
    gray = cv2.medianBlur(gray, 3)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    _, thresh = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(thresh, config=custom_config)

    print("Odczytany tekst:", text.strip())

    cv2.imshow("Znak", roi)
    cv2.imshow("Do OCR", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("❌ Nie znaleziono znaku na obrazie.")
