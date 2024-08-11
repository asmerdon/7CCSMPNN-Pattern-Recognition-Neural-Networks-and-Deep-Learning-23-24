import numpy as np

# Given data
feature_vectors = np.array([[0.0, 2.0], [1.0, 2.0], [2.0, 1.0], [-3.0, 1.0], [-2.0, -1.0], [-3.0, -2.0]])
class_labels = np.array([1, 1, 1, -1, -1, -1])

# Initial parameters
a = np.array([1.0, 0.0, 0.0])
b = np.array([1.0, 0.5, 0.5, 0.5, 0.5, 1.0])
learning_rate = 0.1
epochs = 2

# Function to calculate the updated parameters
def update_parameters(a, y, b, learning_rate):
    aTy = np.dot(a, y)
    a_new = a + learning_rate * (b - aTy) * y
    return aTy, a_new

# Main loop for epochs
for epoch in range(epochs):
    # Iterate over each data point
    for i in range(len(feature_vectors)):
        # Add bias term to feature vector
        x = np.insert(feature_vectors[i], 0, 1)
        y = class_labels[i] * x  # yTk
        aTy, a = update_parameters(a, y, b[i], learning_rate)
        
        # Output values for each iteration
        print(f"Iteration: {epoch * len(feature_vectors) + i + 1}\t aTyk: {aTy:.2f}\t aTnew: {a}")

# Output final parameters after 2 epochs
print("\nFinal Parameters after 2 epochs:")
print("aT: ", a)
