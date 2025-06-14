import pandas as pd

# Load the CSV and treat "\N" as missing values
df = pd.read_csv('titanic.csv', na_values='\\N')

# Optional: Show how many missing values exist per column
print(df.isnull().sum())

# Convert Age to int AFTER filling or dropping NaNs
df['Age'] = df['Age'].fillna(df['Age'].median()).astype(int)

print(df.head())

df.dropna() # Remove rows with missing values
df.fillna(0) # Replace missing values with 0

df.dtypes
df['Age'] = df['Age'].astype(int) # convert if needed

df[df['Age'] > 30]
df[(df['Sex'] == 'male') & (df['Survived']  == 1)]

df.groupby('Sex')['Survived'].mean()
df.groupby(['Pclass', 'Sex']).size()

df.sort_values(by='Age', ascending=False)
