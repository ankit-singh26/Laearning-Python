import numpy as np
import pandas as pd

data = {'cgpa': [8.5, 9.2, 7.8], 'profile_score': [85, 92, 78], 'lpa': [10, 12, 8]}
df = pd.DataFrame(data)
X = df[['cgpa', 'profile_score']].values

def initialize_parameters():
    np.random.seed(1)
    W = np.random.randn(2, 1) * 0.01
    b = np.zeros((1, 1))
    return W, b

def forward_propagation(X, W, b):
    Z = np.dot(X, W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

W, b = initialize_parameters()
A = forward_propagation(X, W, b)
print("Final Output:", A)

