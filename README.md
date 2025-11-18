Group 3 members

MEKONNEN DEMSSIE……………………………………………..DTU14R1247

HERMELA AMARE…………………………………………………DTU14R1554
 
ALEMAYEHU MEKURIA………………………………………….DTU14R1030
 
MUSE DEBALKE……………………………………………………DTU14R1007

DAWIT GEBREMESKEL……………………………………………DTU13R0456
   
RAHEL GETACHEW…………………………………………………DTU14R1343


Diagnosing Heart Disease with Machine Learning
1. Introduction

Heart disease is one of the leading causes of death worldwide. This project aims to predict the presence of heart disease in patients using various health-related features such as age, sex, resting blood pressure, cholesterol, chest pain type, and maximum heart rate achieved.

Machine learning models, specifically Decision Trees and Random Forests, are used to build predictive models that can assist in early risk detection and support medical decision-making.

2. Objectives

To load and understand the Heart Disease dataset.

To clean and preprocess the data for analysis.

To perform Exploratory Data Analysis (EDA) to uncover patterns and relationships.

To build Decision Tree and Random Forest classification models to predict heart disease.

To evaluate the performance of the models using accuracy, precision, recall, F1-score, and confusion matrix.

3. Dataset Description

The dataset used in this project is the Heart Disease Dataset from Kaggle. It contains 303 patient records with multiple clinical features.

Features include:

Age

Sex

Resting blood pressure (trestbps)

Cholesterol level (chol)

Chest pain type (cp)

Maximum heart rate achieved (thalach)

Exercise-induced angina (exang)

Slope of the peak exercise ST segment (slope)

And other clinical measurements

Target variable:

target — 1 indicates presence of heart disease, 0 indicates no heart disease

Dataset link: Kaggle Heart Disease Dataset

4. Tools and Libraries Used

Google Colab

Python 3

Pandas, NumPy — for data manipulation

Matplotlib, Seaborn — for data visualization

Scikit-learn (sklearn) — for modeling and evaluation

Decision Tree & Random Forest classifiers

5. Methodology

The following steps were followed to complete the project:

Data Loading:
Dataset was uploaded to Google Colab and loaded using Pandas (df = pd.read_csv("heart.csv")).

Data Cleaning & Preprocessing:

Checked for missing values and duplicates

Verified data types

Conducted basic statistical analysis

Dataset was found clean with no missing values

Exploratory Data Analysis (EDA):

Plotted correlation heatmaps, age distribution, and chest pain frequency

Observed that chest pain type and maximum heart rate are strong predictors

Feature Engineering:

Split features (X) and target (y)

Applied standard scaling to numerical features

Model Building:

Used Decision Tree and Random Forest classifiers

Train-test split: 80% training, 20% testing

Random Forest was chosen for its robustness and ability to handle nonlinear relationships

Model Evaluation:

Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

Random Forest achieved ~85–90% accuracy

Confusion matrix showed strong separation between classes

6. Results

Key Findings:

Chest pain type (cp) and maximum heart rate achieved (thalach) are the most important predictors of heart disease.

Random Forest outperformed Decision Tree, providing stable predictions.

The small dataset (303 samples) is sufficient for basic modeling but may require expansion for more generalized results.

Sample Confusion Matrix (Random Forest):

[[25, 3],
 [4, 27]]


Feature Importances (Random Forest):

Chest Pain Type (cp): 0.34

Maximum Heart Rate (thalach): 0.28

Age: 0.12

Resting Blood Pressure (trestbps): 0.08

Cholesterol (chol): 0.07

Others: 0.11

7. Conclusion

This project demonstrated a complete workflow for predicting heart disease using machine learning. Data cleaning, EDA, feature engineering, and model building were performed successfully.

Random Forest provided the most reliable predictions, highlighting important features for clinical consideration. This project emphasizes how machine learning can aid in early risk detection for heart disease.

8. References

Kaggle Dataset: Heart Disease Dataset

Scikit-learn Documentation: https://scikit-learn.org/stable/

Python Data Science Tutorials and Resources

9. GitHub Link

GitHub Repository: https://github.com/hermelaamare/-Diagnosing-Heart-Disease-with-Machine-Learning
