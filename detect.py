import cv2
import dlib

# Load the pre-trained HOG + SVM model for face detection
detector = dlib.get_frontal_face_detector()

# Load the pre-trained shape predictor model for face landmarks
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load an image
image = cv2.imread("path_to_your_image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = detector(gray)

# Iterate through detected faces
for face in faces:
    # Get the landmarks/parts for the face in box face
    landmarks = predictor(gray, face)

    # Draw a rectangle around the detected face
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw the facial landmarks
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

# Display the output image
cv2.imshow("Face and Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
