import pandas as pd
import matplotlib.pyplot as plt

# 1. Load CSV dataset
df = pd.read_csv('titanic.csv')

# 2. Display first and last 5 rows
print("First 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())

# 3. Dataset info and summary stats
print("\nDataset information:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())

# 4. Select specific columns
selected_cols = df[['Age', 'Sex', 'Survived']]

# 5. Filter rows (after handling missing values in Age)
df['Age'].fillna(df['Age'].median(), inplace=True)
filtered = df[df['Age'] > 25]

# 6. Use loc and iloc
print("\nUsing loc:")
print(df.loc[0:5, ['Age', 'Sex']])
print("\nUsing iloc:")
print(df.iloc[0:5, 2:5])

# 7. Identify missing values
print("\nMissing values:")
print(df.isnull().sum())

# 8. Fill missing values
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 9. Drop columns with many missing values
df.drop('Cabin', axis=1, inplace=True)

# 10. Remove duplicates
df.drop_duplicates(inplace=True)

# 11. Convert data types
df['Sex'] = df['Sex'].astype('category')

# 12. Normalize column names
df.columns = df.columns.str.lower().str.replace(' ', '_')

# 13. Group by and summary stats
grouped = df.groupby('sex')['survived'].mean()

# 14. Groupby with aggregate functions
agg_df = df.groupby('pclass').agg({'age': 'mean', 'fare': ['sum', 'count']})

# 15. Sort by multiple columns
sorted_df = df.sort_values(['pclass', 'age'], ascending=[True, False])

# 16. Set and reset index
df.set_index('passengerid', inplace=True)
df.reset_index(inplace=True)

# 17. Merge DataFrames
# Create dummy dataframe
classes = pd.DataFrame({
    'pclass': [1, 2, 3],
    'class_type': ['First', 'Second', 'Third']
})
merged = pd.merge(df, classes, on='pclass', how='left')

# 18. Concatenate DataFrames
df1 = df.iloc[:100]
df2 = df.iloc[100:]
vertical_concat = pd.concat([df1, df2])
horizontal_concat = pd.concat([df1, df2], axis=1)

# 19. Basic plots
df['age'].plot.hist(title='Age Distribution')
plt.show()
df.boxplot(column='fare', by='pclass')
plt.show()

# 20. Advanced visualization with Matplotlib
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
df['age'].plot.hist(ax=ax[0], title='Age Distribution')
df['fare'].plot.line(ax=ax[1], title='Fare Distribution')
plt.tight_layout()
plt.show()