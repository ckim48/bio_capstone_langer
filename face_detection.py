# mediapipe --> Open-source library made by google which provides many different
#               advanced AI model.

# Facial landmarks --> specific points on the face that are used to describe features of the face
#                      ex) Points located around the eyes, eyebrows, nose, mouth, ....
# We can choose the number of points on the faces

import cv2 # OpenCV
import mediapipe as mp

# Loading the model which can detect the human face and landmarks.
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()

# Intializes video capture from the webcam
cap = cv2.VideoCapture(0)

# Video --> sequence of images(called frames) 120 MHZ, 60 MHZ
# we keep repeating while the webcam is open/recording
while cap.isOpened():
    # ret --> boolean value --> True if the frame is succsfully captured, false otherwise.
    # frame --> actual image(frame)
    ret, frame = cap.read()
    if not ret: # if the frame is not detected/captured stop the program.
        break
    # Converting the color of the frame from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecting the faces from the frame
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks: # checks if any facial landmarks were detected
        for face_landmarks in results.multi_face_landmarks: # Iterate over every detected facial landmarks
            for i in range(len(face_landmarks.landmark)):
                x = int(face_landmarks.landmark[i].x * frame.shape[1]) # x coordinate of the found landmark
                y = int(face_landmarks.landmark[i].y * frame.shape[0]) # y coordinate of the  found landmark
                cv2.circle(frame, (x,y), 1, (250,0,0), -1) # draw the circle on the landmark
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()