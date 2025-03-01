# Hospital Admissions Prediction Model

This project aims to predict the total number of hospital admissions using a neural network model. The dataset is preprocessed, and a deep learning model is trained to make predictions. The project is implemented in Python using TensorFlow/Keras, Pandas, and Scikit-learn.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)

## Project Overview
The goal of this project is to predict the total number of hospital admissions (`total_admissions`) based on various features in the dataset. The dataset is preprocessed by handling missing values, dropping unnecessary columns, and normalizing the data. A neural network model is then trained to predict the target variable.

## Installation
To run this project, you need to have Python installed along with the following libraries:
- TensorFlow/Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

You can install the required libraries using the following command:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
Usage
Mount Google Drive: The dataset is loaded from Google Drive. Ensure that the file filtered_data.csv is available in your Google Drive.

Preprocessing:

Drop unnecessary columns (total_admissions, Unnamed: 0).

Handle missing values by filling them with the median.

Convert all features to numeric values.

Model Training:

Split the data into training and testing sets.

Define a neural network model with three hidden layers.

Compile the model using the Adam optimizer and mean squared error loss.

Train the model with early stopping to prevent overfitting.

Evaluation:

Evaluate the model on the test set.

Plot the training and validation loss.

Visualize the predicted vs actual admissions.

Results:

Print the test loss and mean squared error (MSE).

Display a scatter plot of predicted vs actual admissions.

Dataset
The dataset used in this project is stored in a CSV file named filtered_data.csv. It contains various features related to hospital admissions, with total_admissions as the target variable.

Model Architecture
The neural network model consists of the following layers:

Input layer with 128 neurons and ReLU activation.

Hidden layer with 64 neurons and ReLU activation.

Hidden layer with 32 neurons and ReLU activation.

Output layer with 1 neuron and linear activation.

The model is compiled using the Adam optimizer with a learning rate of 0.001 and mean squared error as the loss function.

Results
Test Loss: The final test loss after training the model.

Mean Squared Error (MSE): The MSE between the predicted and actual admissions.

Visualization: A scatter plot comparing the predicted vs actual admissions.

License
This project is licensed under the MIT License. See the LICENSE file for details.
