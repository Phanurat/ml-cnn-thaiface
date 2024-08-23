import cv2
import mediapipe as mp

# ตั้งค่า MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# เปิดกล้อง
cap = cv2.VideoCapture(0)

# สร้างตัวตรวจจับมือ
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # แปลงภาพเป็น RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # ตรวจจับมือ
        results = hands.process(rgb_frame)
        
        # วาดการตรวจจับ
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # วาด landmarks ของมือ
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # แสดงผล
        cv2.imshow('Hand Detection', frame)
        
        # ออกจากโปรแกรมเมื่อกด 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ปิดกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
