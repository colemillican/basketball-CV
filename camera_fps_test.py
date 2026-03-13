import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

print(f"Requested: 1280x720 @ 60fps")
print(f"Actual FPS from driver: {cap.get(cv2.CAP_PROP_FPS):.1f}")
print(f"Actual resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print("Measuring throughput...\n")

count = 0
start = time.time()
while count < 300:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        break
    count += 1
    if count % 30 == 0:
        fps = 30 / (time.time() - start)
        print(f"Frames {count-29}-{count}: {fps:.1f} fps")
        start = time.time()

cap.release()
print("\nDone.")
