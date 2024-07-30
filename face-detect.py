import cv2
import numpy as np
from keras.models import load_model

# โหลดโมเดลที่ฝึกไว้
model = load_model('path_to_your_model.h5')

# โหลด Haar Cascade สำหรับการตรวจจับใบหน้า
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# เปิดกล้อง
cap = cv2.VideoCapture(0)

while True:
    # อ่านภาพจากกล้อง
    ret, frame = cap.read()
    if not ret:
        break

    # แปลงภาพเป็น grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับใบหน้า
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # วาดกรอบรอบใบหน้า
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # ตัดส่วนของภาพที่เป็นใบหน้า
        face = frame[y:y+h, x:x+w]

        # แปลงขนาดของใบหน้าให้ตรงกับขนาดที่โมเดลต้องการ
        face = cv2.resize(face, (64, 64))
        face = np.expand_dims(face, axis=0)
        
        # ทำนาย
        prediction = model.predict(face)
        predicted_class = np.argmax(prediction)

        # แสดงผลลัพธ์การทำนาย
        label = f'Province: {predicted_class}'  # ปรับตาม labels จริงของคุณ
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # แสดงภาพที่มีการตรวจจับใบหน้า
    cv2.imshow('Face Detection', frame)

    # กด 'q' เพื่อออกจากลูป
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows()
