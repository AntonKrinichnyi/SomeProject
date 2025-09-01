import numpy as np
import cv2 as cv
import dlib


webcam = cv.VideoCapture(0) # Записуємо відео з вебкамери у змінну
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat") # Завантажуемо модель обличчя
detector = dlib.get_frontal_face_detector() # Встановлюємо детектор обличчя
glasses_image = cv.imread("static/pngegg.png", cv.IMREAD_UNCHANGED) # Завантажуемо зображення окулярів

if glasses_image is None: # Перевірка що завантаження пройшло успішно якщо ні то закриваємо програму
    print("Can't load the file")
    exit()

glasses_alpha = glasses_image[:, :, 3] # Виділяємо альфа канал для зображення окулярів
glasses_rgb = glasses_image[:, :, :3] # Виділяємо rgb канал для зображення окулярів

while True:
    _, frame = webcam.read() # Беремо кожний кадр
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # Перетворюємо кольорове зображення в чорно-біле 
    rects = detector(gray, 0) # Робимо захоплення точок на обличчі

    for rect in rects: # Проходимось по цім точкам циклом
        shape = predictor(gray, rect) # Визначаємо 68 точок на обличчі

        landmarks = np.array([[p.x, p.y] for p in shape.parts()]) # Переводимо точки в numpy массив

        nose_bridge = landmarks[27] # Точка на переносиці для орієнтації по висоті окулярів
        left_eye_outer = landmarks[36] # Точка зовнішньої частини лівого ока для розрахунку ширини окулярів
        right_eye_outer = landmarks[45] # Точка зовнішньої частини правого ока для розрахунку ширини окулярів

        left_eye_center = landmarks[36: 42].mean(axis=0).astype("int") # Центр лівого ока, розраховується за середніми координатами між ключовими точками ока
        right_eye_center = landmarks[42: 48].mean(axis=0).astype("int") # Центр правого ока, розраховується за середніми координатами між ключовими точками ока
        
        dy = right_eye_center[1] - left_eye_center[1] # Розраховуємо різницю координат очей по у 
        dx = right_eye_center[0] - left_eye_center[0] # Розраховуємо різницю координат очей по х

        angle = np.degrees(np.arctan2(dy, dx)) # Розраховуємо кут в радіанах та переводимо їх в градуси
    
        glasses_width = int(np.linalg.norm(right_eye_outer - left_eye_outer) * 1.6) # Розширюємо окуляри

        if glasses_width <= 0: # Перевірка на те щоб ширина була більше 0 інакше програма поверне помилку
            continue

        """
        Змінюємо розмір зображень в rgb та alpha каналах 
        """
        resized_glasses_rgb = cv.resize(glasses_rgb,
                                        (glasses_width,
                                         int(glasses_image.shape[0] * glasses_width / glasses_image.shape[1])),
                                        interpolation=cv.INTER_AREA)
        resized_glasses_alpha = cv.resize(glasses_alpha,
                                          (glasses_width,
                                           int(glasses_image.shape[0] * glasses_width / glasses_image.shape[1])),
                                           interpolation=cv.INTER_AREA)
        
        (h, w) = resized_glasses_rgb.shape[:2] # Дізнаємось де саме на кадрі знаходятся окуляри
        center_of_glasses = (w // 2, h // 2) # Розраховує центр координат для окулярів

        M = cv.getRotationMatrix2D(center_of_glasses, -angle, 1.0) # Повертаємо їх на кут який ми розрахували раніше (кут має бути від'ємним або окуляри будуть повертатись не в ту сторону)

        x_offset = nose_bridge[0] - glasses_width // 2 # Центруємо окуляри по горизонталі та на переносиці
        y_offset = nose_bridge[1] - resized_glasses_rgb.shape[0] // 2 + 5 # Центруємо по вертикалі (+ 5 зміщення вниз, так краще сидять)

        """
        Розраховуємо розмір окулярів для обрізки 
        """
        x1, y1 = max(0, x_offset), max(0, y_offset)
        x2, y2 = min(frame.shape[1], x_offset + resized_glasses_rgb.shape[1]), min(frame.shape[0], y_offset + resized_glasses_rgb.shape[0])
        glasses_roi_x1 = 0
        glasses_roi_y1 = 0
        if x_offset < 0:
            glasses_roi_x1 = -x_offset
        if y_offset < 0:
            glasses_roi_y1 = -y_offset

        overlay_width = x2 - x1 # Ширина області куди будуть вставлятись окуляри
        overlay_height = y2 - y1 # Висота області куди будуть вставлятись окуляри

        if overlay_width <= 0 or overlay_height <= 0: # Валідуємо розмір
            continue # Починаємо наступний цикл якщо розмір не валідний

        """
        Повертаємо зображення в обох каналах
        """
        rotated_glasses_rgb = cv.warpAffine(resized_glasses_rgb, M, (w, h), borderMode=cv.BORDER_REPLICATE)
        rotated_glasses_alpha = cv.warpAffine(resized_glasses_alpha, M, (w, h), borderMode=cv.BORDER_CONSTANT, borderValue=0)

        """
        Власне обрізаємо зображення щоб не отримати помилку виходу за межі массиву
        """
        glasses_overlay_rgb = rotated_glasses_rgb[glasses_roi_y1 : glasses_roi_y1 + overlay_height,
                                                  glasses_roi_x1 : glasses_roi_x1 + overlay_width]
        glasses_overlay_alpha = rotated_glasses_alpha[glasses_roi_y1 : glasses_roi_y1 + overlay_height,
                                                      glasses_roi_x1 : glasses_roi_x1 + overlay_width]

        frame_roi = frame[y1:y2, x1:x2] # Область кадру куди будуть накладатись окуляри

        if glasses_overlay_rgb.shape[:2] != frame_roi.shape[:2]: # Перевіряємо що розміри співпадають
            continue # Починаємо наступний цикл якщо значення не співпадають

        alpha = glasses_overlay_alpha / 255.0 # Вираховуємо альфа-канал
        alpha_inv = 1.0 - alpha # Перетворюємо альфа-канал в float в діапазоні від 0 до 1


        # Наложение: (цвет_фона * (1 - альфа)) + (цвет_переднего_плана * альфа)
        for c in range(0, 3): # Проходимо по кажному каналу (BGR)
            frame_roi[:, :, c] = (frame_roi[:, :, c] * alpha_inv) + \
                                 (glasses_overlay_rgb[:, :, c] * alpha)
            # Накладаємо: (колір фону * (1 - альфа)) + (колір переднього плану * альфа)

    cv.imshow("Output", frame) # Виводимо кожний кадр
    key = cv.waitKey(10) # Чекаємо вводу з клавіатури
    if key == 27: # Пририняємо виконання після натиснення клавіші "esc"
        break

webcam.release() # Відключаємось від вебкамери
cv.destroyAllWindows() # Закриваємо всі вікна
