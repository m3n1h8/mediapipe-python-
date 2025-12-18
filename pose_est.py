import cv2
import mediapipe as mp
import pyttsx3
import numpy as np

engine = pyttsx3.init()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

red_line_y = None

with mp_pose.Pose() as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # ===== KIỂM TRA ĐỘ SÁNG =====
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        if brightness < 90:
            light_status = "Too dark"
            light_color = (0, 0, 255)
        else:
            light_status = "Light OK"
            light_color = (0, 255, 0)

        cv2.putText(frame, f"{light_status} ({brightness:.1f})",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, light_color, 2)

        # ===== POSE =====
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            h, w, _ = frame.shape
            landmarks = result.pose_landmarks.landmark

            left_eye_y = landmarks[2].y * h
            right_eye_y = landmarks[5].y * h
            avg_eye_y = int((left_eye_y + right_eye_y) / 2)

            if cv2.waitKey(1) & 0xFF == ord('o'):
                red_line_y = avg_eye_y
                print("Red line saved at y =", red_line_y)

            if red_line_y is not None:
                cv2.line(frame, (0, red_line_y), (w, red_line_y),
                         (0, 0, 255), 2)

                if avg_eye_y > red_line_y + 10:
                    cv2.putText(frame, "Warning: Stooping posture!",
                                (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 3)
                else:
                    cv2.putText(frame, "Posture OK",
                                (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 3)

        cv2.imshow('Posture Monitor', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
