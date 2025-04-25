# eda.py

# Performs EDA: missing values, encodings, correlations, and plots

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import os

# Load data
df = pd.read_csv('data/titanic_data.csv')

# Create plot directory
os.makedirs('plots', exist_ok=True)

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)

# Drop unneeded columns
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# Encode categorical
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Save cleaned data
df.to_csv('data/cleaned_titanic.csv', index=False)

# Missingno matrix
msno.matrix(df)
plt.title("Missing Values")
plt.savefig("plots/missing_matrix.png")
plt.clf()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("plots/correlation_heatmap.png")
plt.clf()

# Countplot - survival
sns.countplot(data=df, x='Survived')
plt.title("Survival Count")
plt.savefig("plots/survival_distribution.png")
plt.clf()

# Survival by sex
sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival by Sex")
plt.savefig("plots/survival_by_sex.png")
plt.clf()

# Survival by class
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Survival by Pclass")
plt.savefig("plots/survival_by_pclass.png")
plt.clf()

print("✅ EDA Complete. Plots saved in /plots/")
