import numpy as np
import pickle
import pandas as pd
from flask import Blueprint, render_template, request

main = Blueprint('main', __name__)

# Load trained model and preprocessor
model = pickle.load(open('app/model/model.pkl', 'rb'))
preprocessor = pickle.load(open('app/model/preprocess.pkl', 'rb'))

@main.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get values from the form
        features = {
            'Pclass': float(request.form['Pclass']),
            'Sex': request.form['Sex'],  # Keep as string, we will map it
            'Age': float(request.form['Age']),
            'SibSp': float(request.form['SibSp']),
            'Parch': float(request.form['Parch']),
            'Fare': float(request.form['Fare']),
            'Embarked': request.form['Embarked']  # Keep as string, we will map it
        }
        
        # Map 'Sex' and 'Embarked' to numeric values
        features['Sex'] = 0 if features['Sex'].lower() == 'male' else 1
        features['Embarked'] = {'S': 0, 'C': 1, 'Q': 2}.get(features['Embarked'], -1)  # Handle invalid Embarked values
        
        # Convert the features into a DataFrame (same as during training)
        features_df = pd.DataFrame([features])
        
        # Preprocess and scale the features
        features_scaled = preprocessor.transform(features_df)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
    return render_template('index.html', prediction=prediction)
