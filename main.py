# Import Libraries
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

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
print(data)

# remove unnecessary column(id)
data = data.drop('id', axis=1)

# setup columns' name
feature_col = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type',
               'avg_glucose_level', 'bmi', 'smoking_status']
categorical_col = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
target_col = 'stroke'

# encoding categorical data using OrdinalEncoder
for feature in categorical_col:
    feature_set = list(np.unique(data[feature]))
    print(feature_set)
    if feature == 'smoking_status':
        feature_set = ['never smoked', 'formerly smoked', 'smokes']
    elif feature == 'work_type':
        feature_set = ['Children', 'Never_worked', 'Private', 'Govt_job', 'Self-employed']
    feature_raw = {value: index for index, value in enumerate(feature_set)}
    encoder = OrdinalEncoder()
    data[feature] = encoder.fit_transform(data[[feature]])
    print(data.head())

# split feature and target befor scaling
target = pd.DataFrame(data[target_col], columns=[target_col])
features = pd.DataFrame(data.drop(target_col, axis=1), columns=feature_col)
# scaling using RobustScaler
scaler = RobustScaler()
features = pd.DataFrame(scaler.fit_transform(features), columns=feature_col)
target = pd.DataFrame.reset_index(target, drop=True)
data_scaled = pd.concat([features, target], axis=1)
print(data_scaled.head())

# feature selection of each scaled data
# setup feature selection algorithm
k_best = SelectKBest(score_func=f_classif, k=len(feature_col))
extra_tree = ExtraTreesClassifier()
corrmat = data_scaled.corr()
# fitting feature selection model
data_fit = k_best.fit(features, target.values.ravel())
extra_tree.fit(features, target.values.ravel())
# describe best feature
feature_score = pd.DataFrame(pd.concat([pd.DataFrame(features.columns), pd.DataFrame(data_fit.scores_)], axis=1))
feature_score.columns = ['Feature', 'Score']
feature_importance = pd.Series(extra_tree.feature_importances_, index=features.columns)
print(feature_score.nlargest(5, 'Score'))
feature_importance.nlargest(5).plot(kind='barh')
plt.figure(figsize=(20, 20))
gmap = sns.heatmap(data[corrmat.index].corr(), annot=True, cmap="RdYlGn")
plt.show()

# select top 5 feature through mode of every algorithms
feature_selected_col = ['age', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level']

# oversampling for balance target value
smote = SMOTE(random_state=123)
feature_selected = pd.DataFrame(features[feature_selected_col], columns=feature_selected_col)
x, y = smote.fit_resample(feature_selected, target)

# split train and test dataset
train_ratio = 0.2
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=train_ratio, shuffle=True, stratify=y,
                                                    random_state=7)

# model training by Gradient Boosting Classifier
gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train.values.ravel())
pred_init = gbc.predict(x_test)
print("Accuracy of initial model: {0}".format(round(accuracy_score(y_test, pred_init), 3)))

# setup hyper-parameter list for tuning
param_cv = {'n_estimators': range(100, 301, 50),
            'max_depth': range(1, 21),
            'max_features': ['auto', 'sqrt', 'log2'],
            'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'criterion': ['friedman_mse', 'mse']}
# initialize RandomizedSearchCV
rand_cv = RandomizedSearchCV(gbc, param_distributions=param_cv, cv=5, scoring='accuracy', return_train_score=True,
                             n_jobs=-1)
rand_cv.fit(x_train, y_train.values.ravel())
pred_cv = rand_cv.predict(x_test)
print("Accuracy after cross validation: {0}".format(accuracy_score(y_test, pred_cv)))
print("Best parameter: {0}".format(rand_cv.best_params_))
