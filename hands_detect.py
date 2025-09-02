import cv2
import numpy as np
import mediapipe as mp


class HandDetector:
    def __init__(
        self,
        mode=False,
        max_hands=2,
        complexity=1,
        detection_confidance=0.5,
        track_confidance=0.5,
    ):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidance = detection_confidance
        self.track_confidance = track_confidance
        self.complexity = complexity

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            model_complexity=complexity,
            min_detection_confidence=self.detection_confidance,
            min_tracking_confidence=self.track_confidance,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect_hands(self, frame, draw=False):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_image)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        frame, hand_lms, self.mp_hands.HAND_CONNECTIONS
                    )
        return frame
    
    def _hands_coordinate(self, frame, hand_num=0):
        lm_list = []
        if self.results.multi_hand_landmarks:
            hand_lms = self.results.multi_hand_landmarks[hand_num]
            for id, lm in enumerate(hand_lms.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
        return lm_list
    
    def _single_dot_coordinate(self, frame, hand_num=0, dot_id=0):
        id_list = []
        if self.results.multi_hand_landmarks:
            hand_lms = self.results.multi_hand_landmarks[hand_num]
            for id, lm in enumerate(hand_lms.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == dot_id:
                    id_list.append([id, cx, cy])
        return id_list

    def fingen_counter(self, frame):
        thumb = 0
        index_finger = 0
        middle_finger = 0
        ring_finger = 0
        little_finger = 0

        thumb_dot_1 = self._single_dot_coordinate(frame=frame, dot_id=4)
        thumb_dot_2 = self._single_dot_coordinate(frame=frame, dot_id=3)
        if len(thumb_dot_1) != 0 and len(thumb_dot_2) != 0:
            if thumb_dot_1[0][1] - thumb_dot_2[0][1] > 0:
                thumb = 1

        index_finger_dot_1 = self._single_dot_coordinate(frame=frame, dot_id=8)
        index_finger_dot_2 = self._single_dot_coordinate(frame=frame, dot_id=6)
        if len(index_finger_dot_1) != 0 and len(index_finger_dot_2) != 0:
            if index_finger_dot_1[0][2] - index_finger_dot_2[0][2] < 0:
                index_finger = 1

        middle_finger_dot_1 = self._single_dot_coordinate(frame=frame, dot_id=12)
        middle_finger_dot_2 = self._single_dot_coordinate(frame=frame, dot_id=10)
        if len(middle_finger_dot_1) != 0 and len(middle_finger_dot_2) != 0:
            if middle_finger_dot_1[0][2] - middle_finger_dot_2[0][2] < 0:
                middle_finger = 1
        
        ring_finger_dot_1 = self._single_dot_coordinate(frame=frame, dot_id=16)
        ring_finger_dot_2 = self._single_dot_coordinate(frame=frame, dot_id=14)
        if len(ring_finger_dot_1) != 0 and len(ring_finger_dot_2) != 0:
            if ring_finger_dot_1[0][2] - ring_finger_dot_2[0][2] < 0:
                ring_finger = 1

        little_finger_dot_1 = self._single_dot_coordinate(frame=frame, dot_id=20)
        little_finger_dot_2 = self._single_dot_coordinate(frame=frame, dot_id=19)
        if len(little_finger_dot_1) != 0 and len(little_finger_dot_2) != 0:
            if little_finger_dot_1[0][2] - little_finger_dot_2[0][2] < 0:
                little_finger = 1

        return thumb + index_finger + middle_finger + ring_finger + little_finger
