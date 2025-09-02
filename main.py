import numpy as np
import cv2 as cv2

from hands_detect import HandDetector
from wardrobe import glasses_fitting

detector = HandDetector()
webcam = cv2.VideoCapture(0) # Записуємо відео з вебкамери у змінну
while True:
    _, frame = webcam.read() # Беремо кожний кадр
    frame = detector.detect_hands(frame=frame)
    count = detector.fingen_counter(frame=frame)
    frame = glasses_fitting(frame=frame, glasses=count)
    print(count)
    cv2.imshow("Output", frame) # Виводимо кожний кадр
    key = cv2.waitKey(10) # Чекаємо вводу з клавіатури
    if key == 27: # Припиняємо виконання після натиснення клавіші "esc"
        break

webcam.release() # Відключаємось від вебкамери
cv2.destroyAllWindows() # Закриваємо всі вікна
