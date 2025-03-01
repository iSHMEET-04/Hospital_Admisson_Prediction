# Hospital Admissions Prediction Using Random Forest

This project predicts the number of hospital admissions based on features such as arrival month, age, and gender. It uses a **Random Forest Regressor** model to make predictions. The data is preprocessed by encoding categorical variables (month and gender) and scaling numerical features (age). The model is evaluated using Mean Squared Error (MSE).

---

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Results](#results)
- [Prediction Example](#prediction-example)
- [License](#license)

---

## Project Overview
The goal of this project is to predict the number of hospital admissions based on the following features:
- **Arrival Month**: The month of admission (e.g., January, February).
- **Age**: The age of the patient.
- **Gender**: The gender of the patient.

The project uses a **Random Forest Regressor** to model the relationship between these features and the number of admissions. The data is preprocessed using **One-Hot Encoding** for categorical variables and **Standard Scaling** for numerical features.

---

## Installation
To run this project, you need to have Python installed along with the following libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib` (optional, for visualization)

You can install the required libraries using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib
Usage
Load the Dataset: Ensure the dataset is loaded into a Pandas DataFrame (df).

Preprocessing:

Group admissions by month and calculate the total admissions per month.

Encode categorical variables (arrivalmonth and gender) using One-Hot Encoding.

Scale the age feature using Standard Scaling.

Model Training:

Split the data into training and testing sets.

Train a Random Forest Regressor model.

Evaluation:

Evaluate the model using Mean Squared Error (MSE).

Prediction:

Predict admissions for a specific month, age, and gender.

Dataset
The dataset should contain the following columns:

arrivalmonth: The month of admission (e.g., "January", "February").

age: The age of the patient.

gender: The gender of the patient (e.g., "Male", "Female").

admissions: The number of admissions (used for grouping and summing).

Preprocessing
Group Admissions by Month:

The dataset is grouped by arrivalmonth, and the total admissions per month are calculated.

Encode Categorical Variables:

arrivalmonth and gender are encoded using One-Hot Encoding.

Scale Numerical Features:

The age feature is scaled using StandardScaler.

Model Training
A Random Forest Regressor is used to train the model. The model is trained on 80% of the data, and the remaining 20% is used for testing.

python
Copy
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
Results
The model is evaluated using Mean Squared Error (MSE):

python
Copy
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
Prediction Example
You can predict the number of admissions for a specific month, age, and gender. For example:

python
Copy
specific_month = 'January'
specific_age = 28
specific_gender = 'Male'

predicted_admissions = model.predict(input_features)
print(f'Predicted number of admissions for {specific_month}, Age: {specific_age}, Gender: {specific_gender}: {predicted_admissions[0]}')
License
This project is licensed under the MIT License. See the LICENSE file for details.
