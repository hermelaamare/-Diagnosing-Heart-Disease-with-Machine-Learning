# -Diagnosing-Heart-Disease-with-Machine-Learning
: Decision Trees vs. Random Forests.
Heart Disease Prediction Using Machine Learning

A complete data mining project using patient medical data to predict the presence of heart disease.
This project follows a full data-science workflow including data cleaning, EDA, feature engineering, modeling, and evaluation.

📌 Project Objective

The primary goal of this project is to build a machine learning model that can accurately predict whether a patient has heart disease based on health-related features such as:

Age

Sex

Resting blood pressure

Cholesterol level

Chest pain type

Maximum heart rate achieved

Exercise-induced angina

And other clinical measurements

This model can assist in early risk detection and support medical decision-making.

📂 Dataset Information

Dataset Name: Heart Disease Dataset

Source: Kaggle

Format: CSV

Total Records: 303 rows

Target Column: target

1 = Presence of heart disease

0 = No heart disease

🛠 Tools & Technologies Used

Python

Google Colab

Pandas, NumPy – Data manipulation

Matplotlib, Seaborn – Visualization

Scikit-learn – Modeling and evaluation

Random Forest Classifier – Final ML model

📊 Project Workflow (Step-by-Step)
1. Data Loading

Dataset was uploaded to Google Colab and loaded using Pandas.
df = pd.read_csv("heart.csv")

2. Data Cleaning & Preprocessing

Performed:

Missing value identification and handling

Duplicate removal

Data type validation

Outlier checks

Basic statistical summary (df.describe())

The dataset was found to be clean with no missing values.

3. Exploratory Data Analysis (EDA)

Visualizations created:

Correlation Heatmap of all numerical features

Age Distribution Plot

Heart Disease Frequency by Chest Pain Type

Confusion Matrix after model prediction

Important findings:

Chest pain type (cp) strongly correlates with heart disease

Higher age groups show slightly increased risk

Slope (slope) and maximum heart rate (thalach) are important indicators

4. Feature Engineering

Splitting features (X) and target (y)

Standard scaling applied to numerical features using StandardScaler

5. Model Building

Model used:

Random Forest Classifier

Train-test split:

80% training

20% testing

Random Forest was chosen because:

Works well on small/medium datasets

Handles nonlinear relationships

Reduces risk of overfitting

Easy to interpret feature importance

6. Model Evaluation

Metrics used:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Model Results (Your numbers may vary slightly):

Accuracy: ~85–90%

Strong balanced performance overall

Visualization:

Confusion Matrix clearly shows good separation between classes.

📈 Key Insights

Chest pain type and maximum heart rate are strong predictors of heart disease.

Random Forest performed very well and provided stable predictions.

The dataset size is small but sufficient for basic ML modeling.

📉 Model Limitations

Small dataset (303 samples)

No hyperparameter tuning applied

Some features may need deeper medical context analysis

🚀 Future Improvements

Implement GridSearchCV or RandomizedSearchCV

Test additional models (Logistic Regression, XGBoost, SVM)

Add ROC/AUC Curve visualization

Deploy the model using Flask or Streamlit

Expand the dataset for more generalizable performance
