import os

from sklearn import neighbors

from modules import *

NORMALIZE = False

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo="ball_tree", verbose=False, normalize=True):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = load_image_file(img_path)
            win = dlib.image_window(image, class_dir)
            face_bounding_boxes = face_locations(image, win=win, model="hog")

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print(
                        "Image {} not suitable for training: {}".format(
                            img_path,
                            "Didn't find a face"
                            if len(face_bounding_boxes) < 1
                            else "Found more than one face",
                        )
                    )
            else:
                # Add face encoding for current image to the training set
                X.append(face_encodings(image, known_face_locations=face_bounding_boxes, win=win, normalize=normalize)[0])
                y.append(class_dir)


    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)
    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(
        n_neighbors=n_neighbors, algorithm=knn_algo, weights="distance"
    )
    # fit data into knn model
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, "wb") as f:
            pickle.dump(knn_clf, f)

    return knn_clf


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6, normalize=True):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if (
        not os.path.isfile(X_img_path)
        or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS
    ):
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, "rb") as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = load_image_file(X_img_path)
    X_face_locations = face_locations(X_img, model="hog")

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_encodings(X_img, known_face_locations=X_face_locations, normalize=normalize)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    # Mark as False classifications that aren't within the threshold
    are_matches = [
        closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))
    ]

    # Predict classes and remove classifications that aren't within the threshold
    prediction = []
    for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches):
        if rec:
            prediction.append((pred, loc))
        else:
            prediction.append(("unknown", loc))
    return prediction


if __name__ == "__main__":
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    model_name = "trained_knn_model.clf"
    print("Training KNN classifier...")
    classifier = train("./train_faces", model_save_path=model_name, n_neighbors=2, normalize=NORMALIZE)
    print("Training complete!")

    # STEP 2: Using the trained classifier, make predictions for unknown images
    for image_file in os.listdir("./test_faces"):
        full_file_path = os.path.join("./test_faces", image_file)

        print("Looking for faces in {}".format(image_file))

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(full_file_path, model_path=model_name, distance_threshold=0.5, normalize=NORMALIZE)

        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))

        # Display results overlaid on an image
        show_prediction_labels_on_image(os.path.join("./test_faces", image_file), predictions)
