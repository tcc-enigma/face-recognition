# PIPELINE TRAIN

1. An image is selected.
2. The face is detected using dlib.get_frontal_face_detector()
   3. Using the model dlib_face_recognition_resnet_model_v1.dat
3. From the face, 5 or 68 landmarks are found.

![5_landmarks](./docs/5_landmarks.jpg)        ![68_landmarks](./docs/68_landmarks.jpg)

4. Enconding the landmarks, 128 values are generated.
5. These values are used to compose a KNN.
6. KNN is "trained" using the 128 values of each face.
7. KNN uses the ball_tree algorithm and weighted distance.
8. Model is saved.

# PIPELINE TEST