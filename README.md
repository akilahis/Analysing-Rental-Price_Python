# Rental Property Price Prediction

This project focuses on predicting monthly rental prices for properties in Kuala Lumpur and Selangor using machine learning models. The dataset includes features such as property type, size, location, facilities, and proximity to public transport. The goal is to build a predictive model to estimate rental prices accurately.

## Table of Contents
1. [Data Preparation](#data-preparation)
   - [Data Cleaning](#data-cleaning)
   - [Data Transformation](#data-transformation)
2. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
3. [Data Preprocessing](#data-preprocessing)
   - [Feature Encoding](#feature-encoding)
   - [Feature Scaling](#feature-scaling)
4. [Model Development](#model-development)
   - [Model Training](#model-training)
   - [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Results](#results)
6. [Feature Importance](#feature-importance)
7. [Predict with new input](#predict-with-new-input)

---

## Data Preparation

### Data Cleaning
- Loaded the dataset and checked for missing values and duplicates.
- Imputed missing `parking` values using the median for each property type.
- Dropped rows with missing values in critical columns (`completion_year`, `monthly_rent`, etc.).
- Removed outliers using the Interquartile Range (IQR) method for `size` and `monthly_rent`.

### Data Transformation
- Created a new feature `nearby KTM/LRT` to indicate proximity to public transport.
- Binned `completion_year` into decades for better analysis.
- Dropped insignificant columns like `ads_id`.

---

## Exploratory Data Analysis (EDA)
- Visualized the distribution of rental prices by property type, size, and location.
- Analyzed trends in rental prices over decades using bar plots and line charts.
- Created a word cloud to highlight frequently mentioned facilities.
- Examined the impact of furnished status and proximity to public transport on rental prices.

---

## Data Preprocessing

### Feature Encoding
- Encoded categorical variables (`property_type`, `furnished`, `nearby KTM/LRT`, `region`) using one-hot encoding.

### Feature Scaling
- Scaled numerical features using `MinMaxScaler` to normalize the data for modeling.

---

## Model Development

### Model Training
- Trained three models: **Random Forest**, **XGBoost**, and **Decision Tree**.
- Evaluated models using **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)**, and **R² Score**.

### Hyperparameter Tuning
- Optimized **Random Forest** and **XGBoost** using `GridSearchCV` to improve model performance.
- Achieved the best results with:
  - Random Forest: `n_estimators=500`, `max_features='sqrt'`, `min_samples_leaf=1`, `min_samples_split=2`.
  - XGBoost: `n_estimators=500`, `learning_rate=0.1`.

---

## Results
- **Random Forest (Tuned)**: R² = 0.85, MAE = RM 180.
- **XGBoost (Tuned)**: R² = 0.84, MAE = RM 190.
- Visualized predicted vs. actual rental prices to validate model accuracy.

---

## Feature Importance
- Identified key features influencing rental prices:
  - **Top Features**: Property size, location, and furnished status.
  - Visualized feature importance using a bar plot.

---

## Predict with new input
- A tenant is looking to rent a condominium unit of 900sqft, with parking lot and 2 bathrooms with proximity to KTM/LRT in Selangor. What should be the estimated budget?
- The model predicts a monthly rent of RM1125 with error of +/- RM199

![image](https://github.com/user-attachments/assets/4340d736-4b54-47ea-ad9b-78e5693be384)
