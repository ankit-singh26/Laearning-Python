# Accuracy – Overall correctness.

# Precision – What percent of predicted positives are actual positives?
# Precision = TP / (TP + FP)

# Recall – What percent of actual positives are correctly predicted?
# Recall = TP / (TP + FN)

# F1 Score – Harmonic mean of precision and recall.
# F1 = 2 * (Precision * Recall) / (Precision + Recall)

# ROC AUC – Measures model’s ability to distinguish between classes.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print('Logistic Regression Results:')
print(classification_report(y_test, lr_pred, target_names=iris.target_names))

# Model 2: K-Nearest Neighbors
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

print("K-Nearest Neighbors Results:")
print(classification_report(y_test, knn_pred, target_names=iris.target_names))

