import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

from modules import *

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def predict(test_dir):
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
    face_encodings = []
    face_encodings_norm = []
    face_encodings_norm_2 = []
    for class_name in os.listdir(test_dir):
        if not os.path.isdir(os.path.join(test_dir, class_name)):
            continue

        for img_path in image_files_in_folder(os.path.join(test_dir, class_name)):
            face_encodings.append(_get_face_encodigs(img_path, False)[0])
            # face_encodings_norm.append(_get_face_encodigs(img_path, True))
            # face_encodings_norm_2.append(_get_face_encodigs(img_path, True))
    
    plot_encoding(face_encodings)

    for i in range(len(face_encodings_norm)):
        print("Equal: {}".format(np.array_equal(face_encodings_norm[i], face_encodings_norm_2[i])))
        # print("Face encodings for class {}: {}".format(i, face_encodings[i]))
        # print("Face encodings for class {} (normalized): {}".format(i, face_encodings_norm[i]))


def _get_face_encodigs(img_path, normalize):
    # Load image file and find face locations
    X_img = load_image_file(img_path)
    X_face_locations = face_locations(X_img, model="hog")

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    return face_encodings(X_img, known_face_locations=X_face_locations, normalize=normalize)

def plot_encoding(features):
    feature_vector = np.array(features)
    # Apply t-SNE to reduce the dimensionality to 2 dimensions
    tsne = TSNE(n_components=2, perplexity=10, random_state=0)
    reduced_features = tsne.fit_transform(feature_vector)

    # Plot the reduced feature vector
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1])
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization of 128D Feature Vector')
    plt.show()
    plt.waitforbuttonpress()


if __name__ == "__main__":
    # features = [np.random.rand(128) for _ in range(10)]
    # plot_encoding(features)
    test_dir = "./validate_test_faces"
    predict(test_dir)
