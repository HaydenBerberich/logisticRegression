import numpy as np
import matplotlib.pyplot as plt

# Data Preparation
X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
Y = np.array([0, 1, 0, 1, 0, 1, 1, 1])

# Add intercept term
X = np.c_[np.ones(X.shape[0]), X]

# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Log-Likelihood Function
def log_likelihood(X, Y, weights):
    z = np.dot(X, weights)
    ll = np.sum(Y * z - np.log(1 + np.exp(z)))
    return ll

# Gradient Descent
def gradient_descent(X, Y, weights, learning_rate, iterations):
    for i in range(iterations):
        z = np.dot(X, weights)
        predictions = sigmoid(z)
        
        # Gradient of the log-likelihood
        gradient = np.dot(X.T, Y - predictions)
        
        # Update weights
        weights += learning_rate * gradient
    
    return weights

# Prediction Function
def predict(X, weights):
    z = np.dot(X, weights)
    return sigmoid(z)

# Initial weights
weights = np.zeros(X.shape[1])

# Training the model
weights = gradient_descent(X, Y, weights, learning_rate=0.01, iterations=1000)

# Final weights
print("Final weights:", weights)

# Calculate final log-likelihood
final_log_likelihood = log_likelihood(X, Y, weights)
print(f'Final Log-Likelihood: {final_log_likelihood}')

# Calculate accuracy
predictions = predict(X, weights) >= 0.5
accuracy = np.mean(predictions == Y)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Predict the probability of failure for 3 weeks and 5 weeks of not studying
weeks_3 = np.array([1, 3])
weeks_5 = np.array([1, 5])

prob_failure_3 = predict(weeks_3, weights)
prob_failure_5 = predict(weeks_5, weights)

prob_passing_3 = 1 - prob_failure_3
prob_passing_5 = 1 - prob_failure_5

print(f'Probability of passing if not studying for 3 weeks: {prob_passing_3 * 100:.2f}%')
print(f'Probability of passing if not studying for 5 weeks: {prob_passing_5 * 100:.2f}%')

# Points to classify
points_to_classify = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])

# Add intercept term
points_to_classify = np.c_[np.ones(points_to_classify.shape[0]), points_to_classify]

# Predict probabilities
probabilities = predict(points_to_classify, weights)

# Classify based on a threshold of 0.5
classifications = probabilities >= 0.5

# Print results
for point, prob, classification in zip(points_to_classify[:, 1], probabilities, classifications):
    print(f'Point {point}: Probability of failure = {prob:.2f}, Classified as {"Fail" if classification else "Pass"}')