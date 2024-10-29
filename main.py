import threading
import cv2
from deepface import DeepFace

# Open the video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
match_face = False
reference_img = cv2.imread("reference.jpg")

def check_face(frame):
    global match_face
    try:
        # Perform face verification
        result = DeepFace.verify(frame, reference_img, enforce_detection=False)
        match_face = result["verified"]
    except ValueError:
        match_face = False

while True:
    ret, frame = cap.read()
    if ret:
        if counter % 30 == 0:
            # Start a new thread for face verification
            threading.Thread(target=check_face, args=(frame.copy(),)).start()
        counter += 1

        # Display text on the video frame
        if match_face:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NOT MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        # Show the video feed
        cv2.imshow("Video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
