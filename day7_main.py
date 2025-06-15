import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('titanic.csv', na_values='\\N')

# Clean data
df['Age'] = df['Age'].astype(float).fillna(df['Age'].median())
df['Fare'] = df['Fare'].astype(float).fillna(df['Fare'].median())
df['Survived'] = df['Survived'].astype(int)
df['Sex'] = df['Sex'].ffill()
df['Pclass'] = df['Pclass'].astype(int)

# Exploratory Data Analysis(EDA)
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Gender')
plt.show()

sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.show()

g = sns.FacetGrid(df, col='Pclass')
g.map(sns.histplot, 'Age', bins=20)
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df[['Age', 'Fare', 'Survived']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


