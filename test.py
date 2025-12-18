import cv2
import mediapipe as mp
import pyttsx3
engine = pyttsx3.init()
# engine.say("I will speak this text")
# engine.runAndWait()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Biến lưu red line
red_line_y = None

with mp_pose.Pose() as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Lấy tọa độ hai mắt (chuẩn hóa theo chiều ảnh)
            h, w, _ = frame.shape
            landmarks = result.pose_landmarks.landmark

            left_eye_y = landmarks[2].y * h
            right_eye_y = landmarks[5].y * h
            avg_eye_y = int((left_eye_y + right_eye_y) / 2)

            # Khi bấm 'O' => lưu red line
            if cv2.waitKey(1) & 0xFF == ord('o'):
                red_line_y = avg_eye_y
                print("Red line saved at y =", red_line_y)

            # Vẽ red line nếu có
            if red_line_y is not None:
                cv2.line(frame, (0, red_line_y), (w, red_line_y), (0, 0, 255), 2)

                # So sánh vị trí mắt hiện tại với red line
                if avg_eye_y > red_line_y + 10:  # cho phép sai số nhỏ 10px
                    # engine.say("Warning: Stooping posture!")
                    # engine.runAndWait()
                    # engine.stop()
                    cv2.putText(frame, "Warning: Stooping posture!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    cv2.putText(frame, "Posture OK", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow('Posture Monitor', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
