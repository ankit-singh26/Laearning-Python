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

#Line Plot: Age vs Fare
sample = df.sample(50)
plt.plot(sample['Age'], sample['Fare'], marker='o')
plt.title('Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.grid(True)
plt.show()

#Bar Plot:Number of Survivors
survivor_counts = df['Survived'].value_counts()
plt.bar(['Not Survived', 'Survived'], survivor_counts)
plt.title('Survival Count')
plt.ylabel('Count')
plt.show()

#Scatter Plot: Age vd Fare
plt.scatter(df['Age'], df['Fare'], alpha=0.5)
plt.title('Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()

#Histogram: Age distribution
plt.hist(df['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Seaborn Plots
# Boxplot: Age by Survival
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age Distribution by Survival')
plt.xticks([0, 1], ['Not Survived', 'Survived'])
plt.show()

#  Heatmap: Correlation matrix
plt.figure(figsize=(8, 5))
sns.heatmap(df[['Age', 'Fare', 'Survived']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Countplot: Gender distribution
sns.countplot(x='Sex', data=df)
plt.title('Gender Distribution')
plt.show()


