# Adam optimizer
# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target.reshape(-1, 1)  # shape: (20640, 8), (20640, 1)

# Standardize features
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Add bias term
X_scaled = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=42)

# Initialize parameters
theta = np.zeros((X_train.shape[1], 1))  # shape: (9, 1)

# Adam optimizer parameters
alpha = 0.01      # learning rate
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
max_iters = 1000

m = np.zeros_like(theta)
v = np.zeros_like(theta)

costs = []

# Hypothesis
def predict(X, theta):
    return X.dot(theta)

# Cost
def compute_cost(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2) / 2

# Training with Adam
for t in range(1, max_iters + 1):
    y_pred = predict(X_train, theta)
    grad = X_train.T.dot(y_pred - y_train) / X_train.shape[0]
    
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    
    theta -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)
    
    cost = compute_cost(y_pred, y_train)
    costs.append(cost)
    
    if t % 100 == 0:
        print(f"Iteration {t}: Cost = {cost:.4f}")

# Plot loss curve
plt.plot(costs)
plt.title("Training Loss (Adam Optimizer)")
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE / 2)")
plt.grid(True)
plt.show()

# Evaluation
y_pred_test = predict(X_test, theta)
y_pred_test_original = scaler_y.inverse_transform(y_pred_test)
y_test_original = scaler_y.inverse_transform(y_test)

mae = np.mean(np.abs(y_test_original - y_pred_test_original))
print(f"Mean Absolute Error on test set: ${mae * 1000:.2f}")
