import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('titanic.csv', na_values='\\N')

#clean data
df['Age'] = df['Age'].astype(float)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].astype(float)
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Survived'] = df['Survived'].astype(int)
df['Sex'] = df['Sex'].ffill()
df['Pclass'] = df['Pclass'].astype(float)
df['Pclass'] = df['Pclass'].fillna(df['Pclass'].median())

# Count plot
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival Count by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# Facet Grid
g = sns.FacetGrid(df, col='Pclass')
g.map(sns.histplot, 'Age', bins=20, kde=True, color='orange')
plt.suptitle("Age Distribution by Passenger Class", y=1.05)
plt.show()

# Histogram plot
sns.histplot(data = df, x = 'Fare', kde = True)
plt.show()




