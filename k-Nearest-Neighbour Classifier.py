from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import numpy as np

# Load the Iris dataset
iris = datasets.load_iris()

# Given samples
samples = np.array([[7.1, 3.8, 6.7, 2.5],
                    [7.6, 2.0, 2.2, 0.4],
                    [6.2, 3.1, 4.1, 2.4],
                    [7.2, 2.6, 2.3, 0.4],
                    [6.3, 2.7, 4.3, 0.5]])

# Target classes for the samples (ground truth)
ground_truth = np.array([2, 0, 1, 0, 1])

# Create kNN classifiers for k=1 and k=5
knn_k1 = KNeighborsClassifier(n_neighbors=1)
knn_k5 = KNeighborsClassifier(n_neighbors=5)

# Train the classifiers using the Iris dataset
knn_k1.fit(iris.data, iris.target)
knn_k5.fit(iris.data, iris.target)

# Predict the classes for the given samples
predictions_k1 = knn_k1.predict(samples)
predictions_k5 = knn_k5.predict(samples)

# Output the results
print("ANSWER:")
print("For k=1, predicted class for")
for i in range(len(samples)):
    print(f"sample {i + 1}: {predictions_k1[i]}")

print("\nFor k=5, predicted class for")
for i in range(len(samples)):
    print(f"sample {i + 1}: {predictions_k5[i]}")
