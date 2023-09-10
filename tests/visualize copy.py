import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier

# Create a synthetic dataset with 2 features
X, y = make_classification(n_samples=100, n_features=128, random_state=42)

# Create and fit a KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Create a meshgrid for visualization
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Expand the meshgrid to have 128 features (similar to how you expanded your dataset)
expanded_meshgrid = np.random.rand(xx.size, 128)  # Replace with your feature engineering logic

# Use the trained KNeighborsClassifier to make predictions on the expanded meshgrid
Z = knn.predict(expanded_meshgrid)

# Create a contour plot to visualize decision boundaries
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Scatter plot of the original data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KNN Decision Boundaries in 2D')
plt.show()
