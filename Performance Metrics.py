# Given parameters
theta = 0.3
w1 = 0.5
w2 = -0.7

# Feature vectors and true labels
features = [(0.7, -0.6), (0.3, -0.7), (-0.4, -1.0), (0.4, 0.9), (0.5, 0.5), (0.1, -0.5), (-0.5, -0.4), (1.0, -0.6)]
true_labels = [1, 0, 0, 0, 1, 1, 1, 1]

# Function to calculate predicted labels
def predict_label(x):
    output = w1 * x[0] + w2 * x[1] - theta
    return 1 if output >= 0 else 0  # Adjusted the condition

# Create confusion matrix
confusion_matrix = [[0, 0], [0, 0]]

# Display feature vector, true class, and predicted class as a table
print("Feature vector\t\tTrue class\tPredicted class")
for i in range(len(features)):
    true_label = true_labels[i]
    predicted_label = predict_label(features[i])
    print(f"{features[i]}\t{true_label}\t\t{predicted_label}\t\tOutput: {w1 * features[i][0] + w2 * features[i][1] - theta}")

    # Update confusion matrix with corrected indexing
    confusion_matrix[predicted_label][true_label] += 1

# Display confusion matrix
print("\nConfusion Matrix:")
print("\t\tPredicted 1\tPredicted 0")
print(f"True 1\t\t{confusion_matrix[1][1]}\t\t{confusion_matrix[0][0]}")
print(f"True 0\t\t{confusion_matrix[0][1]}\t\t{confusion_matrix[1][0]}")

true_positives = confusion_matrix[1][1]
false_negatives = confusion_matrix[0][0]
false_positives = confusion_matrix[0][1]

print("true pos")
print(true_positives)
print("false pos")
print(false_positives)
print("false neg")
print(false_negatives)

recall = true_positives / (true_positives + false_negatives)
precision = true_positives / (true_positives + false_positives)

# Display results
print(f"\nRecall: {recall:.2f}")
print(f"Precision: {precision:.2f}")
