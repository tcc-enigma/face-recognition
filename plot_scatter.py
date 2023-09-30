import os
from datetime import datetime

import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.decomposition import PCA

from modules import *
from constants import *

def plot(
    face_dir,
):
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
    for class_dir in os.listdir(face_dir):
        if not os.path.isdir(os.path.join(face_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(face_dir, class_dir)):
            image = load_image_file(img_path)
            win = dlib.image_window(image, class_dir)
            face_bounding_boxes = face_locations(image, win=win, model="hog")

            if len(face_bounding_boxes) == 1:
                # Add face encoding for current image to the training set
                X.append(
                    face_encodings(
                        image,
                        known_face_locations=face_bounding_boxes,
                        win=win,
                        normalize=NORMALIZE,
                    )[0]
                )
                y.append(class_dir)

    # Assuming you have a NumPy array X with shape (n_samples, n_features)
    # and a list of class labels y with length n_samples
    # Create a PCA object with 2 components
    pca = PCA(n_components=2)

    # Fit the PCA object to the data and transform the data to the 2-dimensional space
    X_2d = pca.fit_transform(X)

    # Define a colormap to map class labels to colors
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

    # Map each class label to a color
    color_map = dict(zip(np.unique(y), colors))
    c = [color_map[label] for label in y]

    # Plot a scatter of the transformed data with each sample colored according to its class
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=c)

    # Add a legend to the plot
    handles = [
        plt.plot([], [], color=color_map[label], marker="o", ls="", mec="k", mew=0.5, label=label.replace("_", " "))[0]
        for label in np.unique(y)
    ]
    plt.legend(handles=handles, title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
    # Adjust the figure size to include the legend
    # plt.subplots_adjust(right=0.8)

    plt.xlabel("PCA Feature 1")
    plt.ylabel("PCA Feature 2")
    if NORMALIZE:
        normalized = "_normalized"
        title = "KNN Scatter Plot (PCA) Normalized"
    else:
        normalized = ""
        title = "KNN Scatter Plot (PCA)"
    plt.title(title)
    dt_stringnow = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    plt.savefig(
        f"{SAVE_FOLDER}knn_scatter_{dt_stringnow}{normalized}.png", dpi=300, bbox_inches="tight", pad_inches=0.1
    )


if __name__ == "__main__":
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.

    faces = "data/data_faces_from_camera/"

    # create folder plots if not exists
    if not os.path.exists(PLOT_FOLDER):
        os.makedirs(PLOT_FOLDER)
    normalized = "_normalized" if NORMALIZE else ""
    SAVE_FOLDER = f"{PLOT_FOLDER}{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}{normalized}/"
    os.makedirs(SAVE_FOLDER)

    print("Ploting")
    classifier = plot(faces)
    print("Plot Done!")
