import cv2
import numpy as np

img = cv2.imread("data/interim/preprocessing/preprocessed_1787479445.jpg", cv2.IMREAD_GRAYSCALE)
_, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w > 5 and h > 5:
        print(f"Bright object: x={x}, y={y}, w={w}, h={h}")
