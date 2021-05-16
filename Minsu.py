# Import Libraries
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

# Read the dataset
data = pd.read_csv("train_strokes.csv")

# Dataset Check
print("\n************* Data Head ***************")
print(data.head())

print("\n************* Data Description ***************")
print(data.describe())

print("\n************ Data Information ****************")
print(data.info())

##########################################################
# Data Preprocessing

# Cleaning dirty data
# Missing value Check
print("\n************ Check null value ****************")
print(data.isna().sum())

# replace bmi null values
# Split the data to gender (Female, Male, Other)
# replace the mean value of each gender's bmi value in null value
print("\n************ Check gender ****************")
print(data['gender'].value_counts())

# Split the data
female = data['gender'] == 'Female'
male = data['gender'] == 'Male'
other = data['gender'] == 'Other'

data_female = data.copy()[female]
data_male = data.copy()[male]
data_other = data.copy()[other]

# replace the null value using mean
data_female['bmi'].fillna(data_female['bmi'].mean(), inplace=True)
data_male['bmi'].fillna(data_male['bmi'].mean(), inplace=True)
data_other['bmi'].fillna(data_other['bmi'].mean(), inplace=True)

data_bmi = pd.concat([data_female, data_male])
data_bmi = pd.concat([data_bmi, data_other])
data = data_bmi.copy()

# replace smoking_status values
# Split the data to age (under 20 years old, over 20 years old)
# First, people under the age of 20 are filled with 'never smoked'
# Second, another people are filled ffill
print("\n************ Check smoking_status ****************")
print(data['smoking_status'].value_counts())

# Split the data
under20 = data['age'] < 20
over20 = data['age'] >= 20

data_under20 = data.copy()[under20]
data_over20 = data.copy()[over20]

# Replace the null value using ffill
data_under20.fillna('never smoked', inplace=True)
data_over20.fillna(method='ffill', inplace=True)

data_smoking = pd.concat([data_under20, data_over20])
data = data_smoking.copy()

# Cleaning the missing value
print("\n********** Check null (Cleaned value) *************")
print(data.isna().sum())

# Wrong value and Outliers
################### Categorical value #######################
# Check gender
print("\n************ Check gender ****************")
print(data['gender'].value_counts())

# Check hypertension
print("\n************ Check hypertension ****************")
print(data['hypertension'].value_counts())

# Check heart_disease
print("\n************ Check heart_disease ****************")
print(data['heart_disease'].value_counts())

# Check ever_married
print("\n************ Check ever_married ****************")
print(data['ever_married'].value_counts())

# Check work_type
print("\n************ Check work_type ****************")
print(data['work_type'].value_counts())

# Check Residence_type
print("\n************ Check Residence_type ****************")
print(data['Residence_type'].value_counts())

# Check smoking_status
print("\n************ Check smoking_status ****************")
print(data['smoking_status'].value_counts())

# Check stroke
print("\n************ Check stroke ****************")
print(data['stroke'].value_counts())

################# Numeirc value #######################
# Check the outlier using boxplot
# https://medium.com/@stevenewmanphotography/eliminating-outliers-in-python-with-z-scores-dd72ca5d4ead
# Remove Outliers with z-score
# Description = Use the z-score to handle outlier over mean +- 3SD
# Input  = dataframe's column
# Output = index
def find_outliers(col):
    z = np.abs(stats.zscore(col))
    idx_outliers = np.where(z>3, True, False)
    return pd.Series(idx_outliers, index=col.index)

# Check age
# plt.figure(figsize=(5, 5))
# boxplot = data.boxplot(column=['age'])
# plt.show()

# Use z-score to handle outlier
idx_age = find_outliers(data['age'])
data = data.loc[idx_age == False]
print(data.info())

# Check avg_glucose_level
# plt.figure(figsize=(5, 5))
# boxplot = data.boxplot(column=['avg_glucose_level'])
# plt.show()

# Use z-score to handle outlier
idx_avg_glucose = find_outliers(data['avg_glucose_level'])
data = data.loc[idx_avg_glucose == False]
print(data.info())

# Check bmi
# plt.figure(figsize=(5, 5))
# boxplot = data.boxplot(column=['bmi'])
# plt.show()

# Use z-score to handle outlier
idx_bmi = find_outliers(data['bmi'])
data = data.loc[idx_bmi == False]
print(data.info())
print(data)

feature_col = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

for feature in feature_col:
    temp = pd.DataFrame(data[feature])
    data[feature] = pd.get_dummies(temp)
    all_result = pd.get_dummies(temp)
    # result of one-hot encoding
    print("All result: ")
    print(all_result)
    # apply only first columns
    print("Applying to original data")
    print(data[feature])

scaler_mm = preprocessing.MinMaxScaler()
scaler_std = preprocessing.StandardScaler()
scaler_rbs = preprocessing.RobustScaler()
scaler_abs = preprocessing.MaxAbsScaler()
scalers = {'scaler': [scaler_mm, scaler_std, scaler_rbs, scaler_abs]}

for scaler in scalers['scaler']:
    df_scaled = pd.DataFrame(scaler.fit_transform(data))
    print(df_scaled.head())
