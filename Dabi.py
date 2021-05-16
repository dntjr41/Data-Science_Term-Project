# Import Libraries
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
import category_encoders as ce

# Read the dataset
data = pd.read_csv("C:/Users/BEE/Desktop/데이터과학/train_strokes.csv")
"""
# Dataset Check
print("\n************* Data Head ***************")
print(data.head())

print("\n************* Data Description ***************")
print(data.describe())

print("\n************ Data Information ****************")
print(data.info())
"""
##########################################################
# Data Preprocessing

# Cleaning dirty data
# Missing value Check
#print("\n************ Check null value ****************")
#print(data.isna().sum())

# replace bmi null values
# Split the data to gender (Female, Male, Other)
# replace the mean value of each gender's bmi value in null value
#print("\n************ Check gender ****************")
#print(data['gender'].value_counts())

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
#print("\n************ Check smoking_status ****************")
#print(data['smoking_status'].value_counts())

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
#print("\n********** Check null (Cleaned value) *************")
#print(data.isna().sum())

"""
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
"""

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
plt.figure(figsize=(5, 5))
boxplot = data.boxplot(column=['age'])
#plt.show()

# Use z-score to handle outlier
idx_age = find_outliers(data['age'])
data = data.loc[idx_age == False]
#print(data.info())

# Check avg_glucose_level
plt.figure(figsize=(5, 5))
boxplot = data.boxplot(column=['avg_glucose_level'])
#plt.show()

# Use z-score to handle outlier
idx_avg_glucose = find_outliers(data['avg_glucose_level'])
data = data.loc[idx_avg_glucose == False]
#print(data.info())

# Check bmi
plt.figure(figsize=(5, 5))
boxplot = data.boxplot(column=['bmi'])
#plt.show()

# Use z-score to handle outlier
idx_bmi = find_outliers(data['bmi'])
data = data.loc[idx_bmi == False]
#print(data.info())

#print(data.head())


print("\n############ Before encoding ############")
print(data.head())
print(data.info())

# Encoding : OrdinalEncoder
ordinalencoder = OrdinalEncoder()

# gender
# ['Female', 'Male', 'Other'] = [0,1,2]
gender = data[["gender"]]
gender_encod = ordinalencoder.fit_transform(gender)
data['gender']=gender_encod


# ever_married
# ['No', 'Yes'] = [0,1]
married = data[["ever_married"]]
married_encod = ordinalencoder.fit_transform(married)
data['ever_married']=married_encod

# Residence_type
# ['Rural', 'Urban'] = [0,1]
res = data[["Residence_type"]]
res_encod = ordinalencoder.fit_transform(res)
data['Residence_type'] = res_encod


# Ordinal categorical values (smoking_status & work_type)
# smoking_status
# ['never smoked', 'formerly smoked', 'smokes'] = [0,1,2]
# work_type
# ['Children', 'Never_worked', 'Private', 'Govt_job', 'Self-employed'] = [0,1,2,3,4]

ordinal_cols_mapping = [{
    "col": "smoking_status",
    "mapping": {
        'never smoked': 0,
        'formerly smoked': 1,
        'smokes': 2
    }}, {
    "col": "work_type",
    "mapping": {
        'children': 0,
        'Never_worked': 1,
        'Private' : 2,
        'Govt_job' : 3,
        'Self_employed' : 4
    }}
]

ordinal_encoder = ce.OrdinalEncoder(mapping = ordinal_cols_mapping, return_df = True)
data = ordinal_encoder.fit_transform(data)


# float to int
data['gender'] = data['gender'].astype(int)
data['ever_married'] = data['ever_married'].astype(int)
data['work_type'] = data['work_type'].astype(int)
data['Residence_type'] = data['Residence_type'].astype(int)
data['smoking_status']=data['smoking_status'].astype(int)


print("\n\n############ After encoding ############")
print(data.head())
print(data.info())


# Scaling
# Robust Scaling
RobustScaler = preprocessing.RobustScaler()
data_rb = RobustScaler.fit_transform(data)
data_rb = pd.DataFrame(data_rb)
print("\nRobust Scaling")
print(data_rb.head())

# Standard Scaling
StandardScaler = preprocessing.StandardScaler()
data_ss = StandardScaler.fit_transform(data)
data_ss = pd.DataFrame(data_ss)
print("\nStandard Scaling")
print(data_ss.head())

# MinMax Scaling
MinMaxScaler = preprocessing.MinMaxScaler()
data_mm = MinMaxScaler.fit_transform(data)
data_mm = pd.DataFrame(data_mm)
print("\nMinMax Scaling")
print(data_mm.head())

# MaxAbs Scaling
MaxAbsScaler = preprocessing.MaxAbsScaler()
data_ma = MaxAbsScaler.fit_transform(data)
data_ma = pd.DataFrame(data_ma)
print("\nMaxAbs Scaling")
print(data_rb.head())
