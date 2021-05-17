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
plt.figure(figsize=(5, 5))
boxplot = data.boxplot(column=['age'])
plt.show()

# Use z-score to handle outlier
idx_age = find_outliers(data['age'])
data = data.loc[idx_age == False]
print(data.info())

# Check avg_glucose_level
plt.figure(figsize=(5, 5))
boxplot = data.boxplot(column=['avg_glucose_level'])
plt.show()

# Use z-score to handle outlier
idx_avg_glucose = find_outliers(data['avg_glucose_level'])
data = data.loc[idx_avg_glucose == False]
print(data.info())

# Check bmi
plt.figure(figsize=(5, 5))
boxplot = data.boxplot(column=['bmi'])
plt.show()

# Use z-score to handle outlier
idx_bmi = find_outliers(data['bmi'])
data = data.loc[idx_bmi == False]
print(data.info())

########################################################
# Split the feature and target value
# Target = stroke, feature = other
feature_name = ['id', 'gender', 'age', 'hypertension',
                'heart_disease', 'ever_married', 'work_type',
                'Residence_type', 'avg_glucose_level',
                'bmi', 'smoking_status']
data = data.iloc[:, 0:-1]
target = data.iloc[:, -1]

# Encoding : LabelEncoder
labelencoder = preprocessing.LabelEncoder()
data_check = data.copy()

# gender
# ['Female', 'Male', 'Other'] = [0,1,2]
gender = data[["gender"]]
gender_encod = labelencoder.fit_transform(gender)
data['gender']=gender_encod

# Check categorical values
print("\n******************* gender values encoding ******************")
print(data_check['gender'].value_counts())
print(data['gender'].value_counts())

# ever_married
# ['No', 'Yes'] = [0,1]
married = data[["ever_married"]]
married_encod = labelencoder.fit_transform(married)
data['ever_married']=married_encod

# Check categorical values
print("\n******************* ever_married values encoding ******************")
print(data_check['ever_married'].value_counts())
print(data['ever_married'].value_counts())

# work_type
# ['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'children'] = [0,1,2,3,4]
work = data[["work_type"]]
work_encod = labelencoder.fit_transform(work)
data['work_type']=work_encod

# Check categorical values
print("\n******************* work_type values encoding ******************")
print(data_check['work_type'].value_counts())
print(data['work_type'].value_counts())

# Residence_type
# ['Rural', 'Urban'] = [0,1]
res = data[["Residence_type"]]
res_encod = labelencoder.fit_transform(res)
data['Residence_type'] = res_encod

# Check categorical values
print("\n******************* Residence_type values encoding ******************")
print(data_check['Residence_type'].value_counts())
print(data['Residence_type'].value_counts())

# smoking_status
# ['formerly smoked', 'never smoked', 'smokes'] = [0,1,2]
smoke = data[["smoking_status"]]
smoke_encod = labelencoder.fit_transform(smoke)
data['smoking_status'] = smoke_encod

# Check categorical values
print("\n******************* smoking_status values encoding ******************")
print(data_check['smoking_status'].value_counts())
print(data['smoking_status'].value_counts())

# float to int
data['gender'] = data['gender'].astype(int)
data['ever_married'] = data['ever_married'].astype(int)
data['work_type'] = data['work_type'].astype(int)
data['Residence_type'] = data['Residence_type'].astype(int)
data['smoking_status']=data['smoking_status'].astype(int)


print("\n\n############ After encoding ############")
print(data.head())
print(data.info())

###################################################
# Scaling
# Robust Scaling
RobustScaler = preprocessing.RobustScaler()
data_rb = RobustScaler.fit_transform(data)
data_rb = pd.DataFrame(data_rb)
data_rb.columns = feature_name
print("\nRobust Scaling")
print(data_rb.head())

# Standard Scaling
StandardScaler = preprocessing.StandardScaler()
data_ss = StandardScaler.fit_transform(data)
data_ss = pd.DataFrame(data_ss)
data_ss.columns = feature_name
print("\nStandard Scaling")
print(data_ss.head())

# MinMax Scaling
MinMaxScaler = preprocessing.MinMaxScaler()
data_mm = MinMaxScaler.fit_transform(data)
data_mm = pd.DataFrame(data_mm)
data_mm.columns = feature_name
print("\nMinMax Scaling")
print(data_mm.head())

# MaxAbs Scaling
MaxAbsScaler = preprocessing.MaxAbsScaler()
data_ma = MaxAbsScaler.fit_transform(data)
data_ma = pd.DataFrame(data_ma)
data_ma.columns = feature_name
print("\nMaxAbs Scaling")
print(data_rb.head())

##############################################################
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier

# Feature selection - Filter Method.
# SelectKBest
print("\n****** Feature Selection - KBest *******")
# Label encoding + Robust Scaling
np.seterr(divide='ignore', invalid='ignore')
bestfeatures = SelectKBest(score_func=f_classif, k = 11)
KBest_fit = bestfeatures.fit(data_rb, target)
dfcolumns = pd.DataFrame(data_rb.columns)
dfscores = pd.DataFrame(KBest_fit.scores_)

# concatenate two dataframe for better visualization
feature_score = pd.concat([dfcolumns, dfscores], axis = 1)
feature_score.columns = ['Feature', 'Score']
print("\n************* Label + Robust **********")
print(feature_score.nlargest(11, 'Score'))

# Feature selection, drop values
# Drop Residence_type, hypertension, heart_disease, id
data_rb_kbest = data_rb.drop(['Residence_type', 'gender', 'avg_glucose_level', 'id'], axis=1)
print("\n********* Label + Robust *******")
print(data_rb_kbest.info())


# Label encoding + Standard Scaling
bestfeatures = SelectKBest(score_func=f_classif, k = 11)
KBest_fit = bestfeatures.fit(data_ss, target)
dfcolumns = pd.DataFrame(data_ss.columns)
dfscores = pd.DataFrame(KBest_fit.scores_)

# concatenate two dataframe for better visualization
feature_score = pd.concat([dfcolumns, dfscores], axis = 1)
feature_score.columns = ['Feature', 'Score']
print("\n************* Label + Standard **********")
print(feature_score.nlargest(11, 'Score'))

# Feature selection, drop values
# Drop Residence_type, hypertension, heart_disease, id
data_ss_kbest = data_ss.drop(['Residence_type', 'gender', 'avg_glucose_level', 'id'], axis=1)
print("\n********* Label + Standard *******")
print(data_ss_kbest.info())


# Label encoding + MinMax Scaling
bestfeatures = SelectKBest(score_func=f_classif, k = 11)
KBest_fit = bestfeatures.fit(data_mm, target)
dfcolumns = pd.DataFrame(data_mm.columns)
dfscores = pd.DataFrame(KBest_fit.scores_)

# concatenate two dataframe for better visualization
feature_score = pd.concat([dfcolumns, dfscores], axis = 1)
feature_score.columns = ['Feature', 'Score']
print("\n************* Label + MinMax **********")
print(feature_score.nlargest(11, 'Score'))

# Feature selection, drop values
# Drop Residence_type, hypertension, heart_disease, id
data_mm_kbest = data_mm.drop(['Residence_type', 'gender', 'avg_glucose_level', 'id'], axis=1)
print("\n********* Label + MinMax *******")
print(data_mm_kbest.info())


# Label encoding + MaxAbs Scaling
bestfeatures = SelectKBest(score_func=f_classif, k = 11)
KBest_fit = bestfeatures.fit(data_ma, target)
dfcolumns = pd.DataFrame(data_ma.columns)
dfscores = pd.DataFrame(KBest_fit.scores_)

# concatenate two dataframe for better visualization
feature_score = pd.concat([dfcolumns, dfscores], axis = 1)
feature_score.columns = ['Feature', 'Score']
print("\n************* Label + MaxAbs **********")
print(feature_score.nlargest(11, 'Score'))

# Feature selection, drop values
# Drop Residence_type, hypertension, heart_disease, id
data_ma_kbest = data_ma.drop(['Residence_type', 'gender', 'avg_glucose_level', 'id'], axis=1)
print("\n********* Label + MaxAbs *******")
print(data_ma_kbest.info())

# data_rb_kbest, data_ss_kbest, data_mm_kbest, data_ma_kbest

############################################################
# ExtraTreesClassifier
print("\n********** ExtraTreesClassifier**********")
print("********* Cleaned up data *******")
# Label encoding + Robust Scaling
extra_model = ExtraTreesClassifier()
extra_model.fit(data_rb, target)

# Plot graph of feature important
feat_import = pd.Series(extra_model.feature_importances_, index = data_rb.columns)
feat_import.nlargest(11).plot(kind='barh')
plt.show()

# Feature selection, drop values
# Drop Residence_type, hypertension, heart_disease, id
data_rb_extra = data_rb.drop(['Residence_type', 'hypertension', 'heart_disease', 'id'], axis=1)
print("\n********* Label + Robust *******")
print(data_rb_extra.info())


# Label encoding + Standard Scaling
extra_model = ExtraTreesClassifier()
extra_model.fit(data_ss, target)

# Plot graph of feature important
feat_import = pd.Series(extra_model.feature_importances_, index = data_ss.columns)
feat_import.nlargest(11).plot(kind='barh')
plt.show()

# Feature selection, drop values
# Drop Residence_type, hypertension, heart_disease, id
data_ss_extra = data_ss.drop(['Residence_type', 'hypertension', 'heart_disease', 'id'], axis=1)
print("\n********* Label + Standard *******")
print(data_ss_extra.info())


# Label encoding + MinMax Scaling
extra_model = ExtraTreesClassifier()
extra_model.fit(data_mm, target)

# Plot graph of feature important
feat_import = pd.Series(extra_model.feature_importances_, index = data_mm.columns)
feat_import.nlargest(11).plot(kind='barh')
plt.show()

# Feature selection, drop values
# Drop Residence_type, hypertension, heart_disease, id
data_mm_extra = data_mm.drop(['Residence_type', 'hypertension', 'heart_disease', 'id'], axis=1)
print("\n********* Label + MinMax *******")
print(data_mm_extra.info())


# Label encoding + MaxAbs Scaling
extra_model = ExtraTreesClassifier()
extra_model.fit(data_ma, target)

# Plot graph of feature important
feat_import = pd.Series(extra_model.feature_importances_, index = data_ma.columns)
feat_import.nlargest(11).plot(kind='barh')
plt.show()

# Feature selection, drop values
# Drop Residence_type, hypertension, heart_disease, id
data_ma_extra = data_ma.drop(['Residence_type', 'hypertension', 'heart_disease', 'id'], axis=1)
print("\n********* Label + MaxAbs *******")
print(data_ma_extra.info())

# data_rb_extra, data_ss_extra, data_mm_extra, data_ma_extra

# Correlation Matrix with Heatmap