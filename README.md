# Titanic Survival Prediction 🛳️

A Flask-based machine learning project to predict survival of Titanic passengers using a logistic regression classifier.

## Overview
This project leverages the power of machine learning to predict the survival of passengers aboard the Titanic. The web application is built using Flask and serves as a simple interface for users to input the necessary features (such as age, sex, class, etc.) and receive a prediction of whether the passenger survived or not.

## Features
- **Flask Web App UI**: A user-friendly interface to input passenger details and receive a survival prediction.
- **Logistic Regression Classifier**: A logistic regression model trained on historical Titanic data to predict survival.
- **Exploratory Data Analysis (EDA)**: Visualizations for understanding missing data and survival statistics.
- **Preprocessing**: Standard scaling and imputation of missing values before passing data into the model.
- **Model and Preprocessing Saving**: The model and preprocessing steps are saved for reuse in the app.

## Technologies Used
- **Flask**: Python web framework for building the web application.
- **Scikit-Learn**: Machine learning library used for training and testing the model.
- **Pandas**: Data manipulation library used for handling Titanic dataset.
- **Numpy**: Numerical computing library used for feature transformation.
- **Matplotlib/Seaborn**: For visualizing EDA results.


## How to Run

1. Install dependencies
First, ensure you have Python 3.6 or higher installed. Then, create a virtual environment and install the required dependencies:

```bash
# Install dependencies
pip install -r requirements.txt
python eda.py(optional)- for visualization of data and cleaning of data
python train_model.py
python run.py


2.Input Data & Get Predictions
Once the app is running, you can:

Enter the passenger information (Pclass, Sex, Age, etc.) into the form.

Click Predict to get the survival prediction result.

The result will be displayed as either "Survived ✅" or "Did not Survive ❌" based on the prediction made by the model.


#File Descriptions

--train_model.py: Script for training the logistic regression model and saving it for later use.

--run.py: Starts the Flask app.

--app/routes.py: Contains the logic for handling user inputs and making predictions.

--app/templates/index.html: The HTML template for the web interface.

--app/static/style.css: Custom styling for the form.

--data/titanic_data.csv: The dataset used to train the model (Titanic dataset).

-Example of Prediction Input
--Here’s an example of what a typical input might look like:

-Pclass: 1 (First Class)

-Sex: Female

-Age: 22

-SibSp: 1 (1 sibling/spouse aboard)

-Parch: 0 (No parents/children aboard)

-Fare: 71.2833

-Embarked: C (Cherbourg)

-Prediction Example:
If a passenger with these details is predicted to survive, the message displayed will be:

-<h3>Prediction: Survived ✅</h3>
If the prediction is that they did not survive:

-<h3>Prediction: Did not Survive ❌</h3>



I also attached screenshot of input and output , it will surely help you out