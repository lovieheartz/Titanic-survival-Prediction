# train_model.py

# Trains a logistic regression model and saves it to app/model/.

import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Load the Titanic dataset
df = pd.read_csv('data/titanic_data.csv')

# Feature selection and target
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

# Encode categorical values
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
X['Embarked'] = X['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline for numeric data
numeric_features = X.columns.tolist()
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])

# Full pipeline with logistic regression
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Train and evaluate
model_pipeline.fit(X_train, y_train)
accuracy = model_pipeline.score(X_test, y_test)
print(f"✅ Model trained. Accuracy: {accuracy:.4f}")

# Save model and preprocessor
os.makedirs('app/model', exist_ok=True)
pickle.dump(model_pipeline.named_steps['classifier'], open('app/model/model.pkl', 'wb'))
pickle.dump(model_pipeline.named_steps['preprocessor'], open('app/model/preprocess.pkl', 'wb'))
