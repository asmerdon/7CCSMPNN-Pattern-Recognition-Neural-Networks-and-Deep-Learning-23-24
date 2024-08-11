import numpy as np

# Given data
feature_vectors = np.array([[0.0, 2.0],
                            [1.0, 2.0],
                            [2.0, 1.0],
                            [-3.0, 1.0],
                            [-2.0, -1.0],
                            [-3.0, -2.0]])

class_labels = np.array([1, 1, 1, 0, 0, 0])

# Initial parameters
theta = -0.5
w1 = -4.5
w2 = -1.5
learning_rate = 1.0
epochs = 2

# Heaviside function
def heaviside_function(z):
    return 1.0 if z >= 0 else 0.0

# Function to flip the sign of theta and print a message
def flip_theta_and_print_message(theta):
    new_theta = -theta
    print("CHECK SIGN OF THETA!!! This code is a bit hacky and may not give correct output if not checked properly.")
    return new_theta

# Flip the sign of theta and print a message after each epoch
theta = flip_theta_and_print_message(theta)

# Main loop for epochs
for epoch in range(epochs):
    # Iterate over each data point
    for i in range(len(feature_vectors)):
        # Extract features and add bias term
        x = np.insert(feature_vectors[i], 0, 1)
        
        # Calculate the weighted sum and apply the Heaviside function
        wx = np.dot(np.array([theta, w1, w2]), x)
        y = heaviside_function(wx)
        
        # Update weights
        w_new = np.array([theta, w1, w2]) + learning_rate * (class_labels[i] - y) * x
        
        # Output values for each iteration
        print(f"Iteration: {epoch * len(feature_vectors) + i + 1}\t y=H(wxk): {y:.2f}\t wnew: {tuple(w_new)}")
        
        # Update weights for the next iteration
        theta, w1, w2 = w_new
    
