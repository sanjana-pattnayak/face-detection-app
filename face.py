import cv2

# Load Haar Cascade from OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not working")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30)
    )

    # Draw rectangle and label for each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

        cv2.putText(frame, "Face", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Show detection status
    if len(faces) > 0:
        cv2.putText(frame, "Face Detected!", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(frame, "No Face Detected", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Show face count
    cv2.putText(frame, f'Faces: {len(faces)}', (10,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    # Display output
    cv2.imshow("Face Detection App", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()