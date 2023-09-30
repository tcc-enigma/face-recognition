import os
import dlib
import csv
import numpy as np
import logging
import cv2

# Path of cropped faces
PATH_CAMERA_FACE = "data/data_faces_from_camera/"
PATH_CSV = "data/faces_features.csv"

# Use frontal face detector of Dlib
face_detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")

# Get face landmarks
pose_predictor_68_point = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
# pose_predictor_5_point = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")

# Use Dlib resnet50 model to get 128D face descriptor
face_encoder = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")


# Return 128D features for single image
# Input:    path_img           <class 'str'>
# Output:   face_descriptor    <class 'dlib.vector'>
def return_128d_features(path_img):
    img_rd = cv2.imread(path_img)
    faces = face_detector(img_rd, 1)

    logging.info("%-40s %-20s", "Image with faces detected:", path_img)

    # For photos of faces saved, we need to make sure that we can detect faces from the cropped images
    if len(faces) != 0:
        shape = pose_predictor_68_point(img_rd, faces[0])
        face_descriptor = face_encoder.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
        logging.warning("no face")
    return face_descriptor


# Return the mean value of 128D face descriptor for person X
# Input:    path_face_personX        <class 'str'>
# Output:   features_mean_personX    <class 'numpy.ndarray'>
def return_features_mean_person(path_face_person):
    features_list_person = []
    photos_list = os.listdir(path_face_person)
    if photos_list:
        for i in range(len(photos_list)):
            # Get 128D features for single image of personX
            logging.info("%-40s %-20s", "Reading image:", path_face_person + "/" + photos_list[i])
            features_128d = return_128d_features(path_face_person + "/" + photos_list[i])
            if features_128d:
                features_list_person.append(features_128d)
    else:
        logging.warning("Warning: No images in %s/", path_face_person)

    # Compute the mean
    if features_list_person:
        # Original code
        # features_mean_personX = np.array(features_list_person, dtype=object).mean(axis=0)

        features_mean_person = np.mean(features_list_person, axis=0, dtype=object)
    else:
        features_mean_person = np.zeros(128, dtype=object, order="C")
    return features_mean_person


def main():
    logging.basicConfig(level=logging.INFO)
    # Get the order of latest person
    faces_folder = os.listdir(PATH_CAMERA_FACE)
    faces_folder.sort()

    with open(PATH_CSV, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        for person in faces_folder:
            # Get the mean/average features of face/personX, it will be a list with a length of 128D
            # logging.info("%sperson_%s", PATH_CAMERA_FACE, person)
            features_mean_personX = return_features_mean_person(PATH_CAMERA_FACE + person)
            features_mean_personX = np.insert(features_mean_personX, 0, person, axis=0)
            # features_mean_personX will be 129D, person name + 128 features
            writer.writerow(features_mean_personX)
            logging.info("\n")

        logging.info(f"Save all the features of faces registered into: {PATH_CSV}")


if __name__ == "__main__":
    main()
