Benchmark for face verification
[http://vis-www.cs.umass.edu/lfw/#resources](http://vis-www.cs.umass.edu/lfw/#resources)

PIPELINE TRAIN

1. An image is selected.
2. The face is detected using HOG or CNN

   * **HOG**: [http://dlib.net/python/index.html#dlib_pybind11.get_frontal_face_detector](http://dlib.net/python/index.html#dlib_pybind11.get_frontal_face_detector)
   * **CNN**:
3. Use model to get landmarks

   * **5** **landmarks:** [https://github.com/davisking/dlib-models#shape_predictor_5_face_landmarksdatbz2](https://github.com/davisking/dlib-models#shape_predictor_5_face_landmarksdatbz2)
   * **68 landmarks:** [https://github.com/davisking/dlib-models#shape_predictor_68_face_landmarksdatbz2](https://github.com/davisking/dlib-models#shape_predictor_68_face_landmarksdatbz2)

![5_landmarks](./docs/5_landmarks.jpg)        ![68_landmarks](./docs/68_landmarks.jpg)

4. Enconding the landmarks, 128 values are generated.
   It should also be noted that you can also call this function like this:
   face_descriptor = facerec.compute_face_descriptor(img, shape, 100, 0.25)
   The version of the call without the 100 gets 99.13% accuracy on LFW
   while the version with 100 gets 99.38%.  However, the 100 makes the
   call 100x slower to execute, so choose whatever version you like.  To
   explain a little, the 3rd argument tells the code how many times to
   jitter/resample the image.  When you set it to 100 it executes the
   face descriptor extraction 100 times on slightly modified versions of
   the face and returns the average result.  You could also pick a more
   middle value, such as 10, which is only 10x slower but still gets an
   LFW accuracy of 99.3%.
   4th value (0.25) is padding around the face. If padding == 0 then the chip will
   be closely cropped around the face. Setting larger padding values will result a looser cropping.
   In particular, a padding of 0.5 would double the width of the cropped area, a value of 1. would triple it, and so forth.
5. These values are used to compose a KNN.
6. KNN is "trained" using the 128 values of each face.
7. KNN uses the ball_tree algorithm and weighted distance.
   * [https://scikit-learn.org/stable/modules/neighbors.html](https://scikit-learn.org/stable/modules/neighbors.html)
   * Neighbors-based classification is a type of *instance-based learning* or  *non-generalizing learning* : it does not attempt to construct a general internal model, but simply stores instances of the training data. Classification is computed from a simple majority vote of the nearest neighbors of each point: a query point is assigned the data class which has the most representatives within the nearest neighbors of the point.
   * Where KD trees partition data along Cartesian axes, **ball trees** partition data in a series of nesting hyper-spheres. This makes tree construction more costly than that of the KD tree, but results in a data structure which can be very efficient on highly structured data, even in very high dimensions.
   * [https://scikit-learn.org/stable/modules/neighbors.html#ball-tree](https://scikit-learn.org/stable/modules/neighbors.html#ball-tree)
   * [https://citeseerx.ist.psu.edu/doc_view/pid/17ac002939f8e950ffb32ec4dc8e86bdd8cb5ff1](https://citeseerx.ist.psu.edu/doc_view/pid/17ac002939f8e950ffb32ec4dc8e86bdd8cb5ff1)
8. The model is saved.

PIPELINE TEST

1. An image is selected.
2. The face is detected using `dlib.get_frontal_face_detector()`.
3. Using the model `dlib_face_recognition_resnet_model_v1.dat`, 5 or 68 landmarks are found from the face.
4. The KNN model is used to find the best matches for the test face.
5. Either classification is selected as True or False using the threshold.
6. The prediction is shown.

---

Matrix confusão
