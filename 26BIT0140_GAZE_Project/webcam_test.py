
import cv2

# Open the laptop's default webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam started. Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read a camera frame.")
            break

        cv2.imshow("My Webcam - Gaze Project", frame)

        # Press Q to close the webcam window
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()