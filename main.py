import numpy as np
import cv2 as cv2
from wardrobe import glasses_fitting


webcam = cv2.VideoCapture(0) # Записуємо відео з вебкамери у змінну
while True:
    _, frame = webcam.read() # Беремо кожний кадр
    frame = glasses_fitting(frame=frame)
    cv2.imshow("Output", frame) # Виводимо кожний кадр
    key = cv2.waitKey(10) # Чекаємо вводу з клавіатури
    if key == 27: # Припиняємо виконання після натиснення клавіші "esc"
        break

webcam.release() # Відключаємось від вебкамери
cv2.destroyAllWindows() # Закриваємо всі вікна
