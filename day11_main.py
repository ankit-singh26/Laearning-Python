# Decision Tree
# A flowchart-like tree structure.
# Easy to interpret, can overfit.

# K-Nearest Neighbors
# Classifies based on the majority class of the k nearest data points.

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision tree
dt_model = DecisionTreeClassifier(max_depth=5)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

print("Decision Tree Classifier Report:")
print(classification_report(y_test, dt_preds, target_names=data.target_names))

# KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_preds = knn_model.predict(X_test)

print("K-Nearest Neighbors Classifier Report:")
print(classification_report(y_test, knn_preds, target_names=data.target_names))

# Confusion Matrix for visualization
cm_dt = confusion_matrix(y_test, dt_preds)
cm_knn = confusion_matrix(y_test, knn_preds)

disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=data.target_names)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=data.target_names)

disp1.plot(cmap='Blues')
disp2.plot(cmap='Greens')

feat_importances = pd.DataFrame(
    dt_model.feature_importances_,
    index=data.feature_names,
    columns=["Importance"]
)
feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
feat_importances.plot(kind='bar', figsize=(10, 6))
plt.title("Feature Importances (Decision Tree)")
plt.tight_layout()
plt.show()
