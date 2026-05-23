# Task 5 — Personal Loan Acceptance Prediction

## Project Overview

This project focuses on predicting whether customers are likely to accept a personal loan offer using Machine Learning techniques. The Bank Marketing dataset was analyzed, cleaned, visualized, and used to train a Logistic Regression model for classification.

The project demonstrates the complete Machine Learning workflow including:

* Data loading
* Data preprocessing
* Exploratory Data Analysis (EDA)
* Feature encoding
* Feature scaling
* Model training
* Model evaluation
* Prediction analysis

---

# Objective

The main objective of this project is to identify customers who are more likely to accept a personal loan offer based on customer demographic and banking-related information.

---

# Technologies and Libraries Used

The following tools and libraries were used in this project:

* Python
* pandas
* matplotlib
* seaborn
* scikit-learn
* Jupyter Notebook

---

# Dataset Information

Dataset Name:

`bank.csv`

The dataset contains customer information such as:

* Age
* Job
* Marital Status
* Education
* Balance
* Housing Loan
* Personal Loan
* Contact Type
* Campaign Information
* Previous Marketing Outcome
* Deposit Subscription Status

Target Variable:

`deposit`

* Yes → Customer accepted the offer
* No → Customer did not accept the offer

---

# Project Workflow

## 1. Loading the Dataset

The dataset was loaded using pandas.

```python
import pandas as pd

df = pd.read_csv("bank.csv")
```

---

## 2. Dataset Exploration

Initial exploration was performed to understand:

* Dataset structure
* Column names
* Data types
* Number of rows and columns
* Sample records

Functions used:

```python
df.head()
df.info()
df.shape
df.columns
```

---

## 3. Checking Missing Values

The dataset was checked for missing values.

```python
df.isnull().sum()
```

---

## 4. Data Preprocessing

Machine Learning models cannot work directly with text data.

Categorical columns were converted into numerical values using Label Encoding.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
```

A separate copy of the dataset was created for Machine Learning operations.

---

## 5. Exploratory Data Analysis (EDA)

Different visualizations were created to understand customer behavior.

### Age Distribution

A histogram was used to analyze customer age distribution.

### Marital Status Analysis

A count plot was used to compare different marital groups.

### Job Category Analysis

A count plot was used to analyze customer job categories.

Libraries used:

```python
import seaborn as sns
import matplotlib.pyplot as plt
```

---

## 6. Feature and Target Selection

Features and target variable were separated.

```python
X = df_ml.drop("deposit", axis=1)
y = df_ml["deposit"]
```

---

## 7. Splitting the Dataset

The dataset was divided into training and testing sets.

```python
from sklearn.model_selection import train_test_split
```

* 80% Training Data
* 20% Testing Data

---

## 8. Feature Scaling

Feature scaling was applied using StandardScaler to improve Logistic Regression performance.

```python
from sklearn.preprocessing import StandardScaler
```

---

## 9. Model Training

A Logistic Regression model was trained on the dataset.

```python
from sklearn.linear_model import LogisticRegression
```

The model was trained using:

```python
model = LogisticRegression(max_iter=5000)
```

---

## 10. Model Prediction

Predictions were generated using the trained model.

```python
predictions = model.predict(X_test)
```

---

## 11. Model Evaluation

The model performance was evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report

### Accuracy Score

Measures the overall correctness of predictions.

### Confusion Matrix

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

### Classification Report

Provides:

* Precision
* Recall
* F1-Score

---

# Key Insights

* Customer age influences loan acceptance behavior.
* Marital status impacts customer response patterns.
* Job categories affect decision-making.
* Logistic Regression can successfully classify customer responses.
* Feature scaling improved model training performance.

---

# Conclusion

This project successfully implemented a Machine Learning model for predicting customer loan acceptance using the Bank Marketing dataset.

The project covered the complete workflow from data preprocessing and visualization to model training and evaluation.

Logistic Regression achieved effective classification results and demonstrated how customer demographic and financial information can be used for predictive analytics.



# Author

Amna Umer

BS Computer Science
