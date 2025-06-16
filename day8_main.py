from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Generate a synthetic regression dataset
X, y = make_regression(n_samples=100 , n_features=1, noise=10, random_state=42)

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\

# Step 3: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Predict on the test set
y_pred =  model.predict(X_test)

# Step 5: Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
print("Coefficient (slope):", model.coef_[0])
print("Intercept:", model.intercept_)

# Step 6: Visualize the data and the regression line
plt.figure(figsize=(10,6))
sns.scatterplot(x=X_test.flatten(), y=y_test, label='Actual', color='blue')
sns.lineplot(x=X_test.flatten(), y=y_pred, label='Predicted', color='red')
plt.title('Linear Regression Fit')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.grid(True)
plt.show()

# Step 7: Residuals plot (how far off predictions are)
residuals = y_test - y_pred
plt.figure(figsize=(8,5))
sns.histplot(residuals, kde=True, color='purple')
plt.title("Distribution of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()