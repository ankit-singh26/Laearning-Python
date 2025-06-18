import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the California housing dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target.reshape(-1, 1)  # Reshape for consistency

# We'll use only one feature for visualization, e.g., 'MedInc' (median income)
X = X[:, [0]]  # Column 0 is 'MedInc'

# Standardize the feature
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SGD Function
def sgd(X, y, learning_rate=0.1, epochs=500, batch_size=1):
    m = len(X)
    theta = np.random.randn(2, 1)
    X_bias = np.c_[np.ones((m, 1)), X]
    cost_history = []

    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X_bias[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            gradients = 2 / batch_size * X_batch.T.dot(X_batch.dot(theta) - y_batch)
            theta -= learning_rate * gradients

        predictions = X_bias.dot(theta)
        cost = np.mean((predictions - y) ** 2)
        cost_history.append(cost)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Cost: {cost:.4f}")

    return theta, cost_history

# Train the model using SGD
theta_final, cost_history = sgd(X_train, y_train, learning_rate=0.1, epochs=500, batch_size=1)

# Plot cost function
plt.plot(cost_history)
plt.xlabel('Epochs')
plt.ylabel('Cost (MSE)')
plt.title('Cost Function during Training')
plt.grid(True)
plt.show()

# Plot predictions
X_plot = np.c_[np.ones((X_test.shape[0], 1)), X_test]
y_pred = X_plot.dot(theta_final)

plt.scatter(X_test, y_test, color='blue', label='Actual prices')
plt.plot(X_test, y_pred, color='red', label='Predicted line')
plt.xlabel('Median Income (Standardized)')
plt.ylabel('House Price ($100,000s)')
plt.title('SGD Linear Regression on Real Housing Data')
plt.legend()
plt.grid(True)
plt.show()

print(f"Final parameters (theta): {theta_final}")
