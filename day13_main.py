# 1. Load & explore
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('titanic.csv', na_values='\\N')
print(df.head())
print(df.info())
print(df.isnull().sum())

# Visualize survival counts
sns.countplot(x='Survived', data=df)
plt.show()

# 2. Data Cleaning & Feature Engineering
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# 3. Train-Test Split
from sklearn.model_selection import train_test_split

features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Embarked_Q', 'Embarked_S']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build Model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate Model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Optional: Feature Importance
importances = pd.Series(model.feature_importances_, index=features)
importances.plot(kind='barh')
plt.title('Feature Importances')
plt.show()