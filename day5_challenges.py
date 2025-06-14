import pandas as pd
import seaborn as sns

# Load dataset
df = sns.load_dataset('titanic')

# 1. Preview
print(df.head())
print(df.info())
print(df.describe())

# 2. Cleaning
df = df.dropna(subset=['age'])  # remove rows where age is missing

# 3. Filtering
print(df[df['survived'] == 1])  # survivors

# 4. Grouping
print(df.groupby('sex')['survived'].mean())  # survival rate by gender

# 5. Sorting
print(df.sort_values(by='age', ascending=False).head(10))  # oldest 10 passengers

df.to_csv('cleaned_titanic.csv', index=False)
