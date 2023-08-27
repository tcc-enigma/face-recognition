import cv2
import dlib

# Load the image
img = cv2.imread('test_faces\\billiehd.jpg')

# Initialize the detector and predictor
detector = dlib.get_frontal_face_detector()
n_landmarks = 68
predictor = dlib.shape_predictor(f'models\\shape_predictor_{n_landmarks}_face_landmarks.dat')

# Detect faces in the image
faces = detector(img)
# Loop through each face and draw landmarks
for face in faces:
    # Get the landmarks/parts for the face in box d.
    landmarks = predictor(img, face)
    
    # Loop through all the points and draw them on the image
    for n in range(0, n_landmarks):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        
        cv2.putText(img, f'{n}', (x - 8, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (188, 73, 73), 2)
        cv2.circle(img, (x, y), 2, (0, 255, 0), 2)

# Save the image with landmarks
cv2.imwrite(f'{n_landmarks}_landmarks.jpg', img)   

# Show the image with landmarks
cv2.imshow('Image with Landmarks', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
