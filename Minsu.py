# Import Libraries
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV

# Read the dataset
from sklearn.neighbors import KNeighborsClassifier

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

# remove unnecessary column(id)
data = data.drop('id', axis=1)

# setup features' column name to encode
feature_col_encode = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# one-hot encoding each feature
for feature in feature_col_encode:
    temp = pd.DataFrame(data[feature])
    data[feature] = pd.get_dummies(temp)
    all_result = pd.get_dummies(temp)
    # result of one-hot encoding
    print("All result: ")
    print(all_result)
    # apply only first columns
    print("Applying to original data")
    print(data[feature])

# setup columns' name
feature_col = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type',
               'avg_glucose_level', 'bmi', 'smoking_status']
target_col = 'stroke'

# separate feature and target
target = data[target_col]
features = pd.DataFrame(data.drop(target_col, axis=1), columns=feature_col)
print("The number of features: {0}".format(len(features)))
print("The number of target: {0}".format(len(target)))

# scaler setup
scaler_mm = preprocessing.MinMaxScaler()
scaler_std = preprocessing.StandardScaler()
scaler_rbs = preprocessing.RobustScaler()
scaler_abs = preprocessing.MaxAbsScaler()
scalers = {'scaler': [scaler_mm, scaler_std, scaler_rbs, scaler_abs]}
scaler_name = ['MinMax', 'Standard', 'Robust', 'MaxAbs']

# scaling by each method
data_scaled = list()
for scaler in scalers['scaler']:
    df_scaled = pd.DataFrame(scaler.fit_transform(features), columns=feature_col)
    data_scaled.append(df_scaled)
    print(df_scaled.head())

# feature selection of each scaled data
for i, dataset in enumerate(data_scaled):
    # setup feature selection algorithm
    k_best = SelectKBest(score_func=mutual_info_classif, k=len(feature_col))
    extra_tree = ExtraTreesRegressor()
    corrmat = dataset.corr()
    # fitting feature selection model
    data_fit = k_best.fit(dataset, target)
    extra_tree.fit(dataset, target)
    # describe best feature
    feature_score = pd.DataFrame(pd.concat([pd.DataFrame(dataset.columns), pd.DataFrame(data_fit.scores_)], axis=1))
    feature_score.columns = ['Feature', 'Score']
    feature_importance = pd.Series(extra_tree.feature_importances_, index=dataset.columns)
    print(feature_score.nlargest(5, 'Score'))
    feature_importance.nlargest(5).plot(kind='barh')
    plt.figure(figsize=(20, 20))
    gmap = sns.heatmap(data[corrmat.index].corr(), annot=True, cmap="RdYlGn")
    # plt.show()

# drop unselected feature and sampling
features_sampled = list()
target_sampled = list()
feature_selected = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'smoking_status']
for scaled in data_scaled:
    smote = SMOTE(random_state=11)
    # extract top 5 most selected features
    scaled = pd.DataFrame(scaled[feature_selected], columns=feature_selected)
    # resampling to solve unbalance problem
    x, y = smote.fit_resample(scaled, target)
    features_sampled.append(x)
    target_sampled.append(y)

valid_ratio = 0.2
# split train and test dataset
set_x_train = list()
set_x_test = list()
set_y_train = list()
set_y_test = list()
for n in range(len(features_sampled)):
    x_data = features_sampled[n]
    y_data = target_sampled[n]
    # split train and test data
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=valid_ratio, shuffle=True,
                                                        stratify=y_data, random_state=123)
    # store in dataset list
    set_x_train.append(x_train)
    set_x_test.append(x_test)
    set_y_train.append(y_train)
    set_y_test.append(y_test)

# fitting all model
for i in range(len(set_x_train)):
    # get train data each scaled
    x_temp = set_x_train[i]
    y_temp = set_y_train[i]
    # setup classifier models
    reg = LogisticRegression(solver='saga')
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k)
    e = 5
    ratio = 0.8
    gbc = GradientBoostingClassifier(learning_rate=ratio, n_estimators=e)
    # fit operation
    reg.fit(x_temp, y_temp)
    knn.fit(x_temp, y_temp)
    gbc.fit(x_temp, y_temp)
    # prediction to check accuracy of model
    test_x = set_x_test[i]
    test_y = set_y_test[i]
    pred_r = reg.predict(test_x)
    pred_k = knn.predict(test_x)
    pred_g = gbc.predict(test_x)
    # show accuracy (first version)
    print("Accuracy of Logistic Regression: {0}".format(accuracy_score(test_y, pred_r)))
    print("Accuracy of KNN: {0}".format(accuracy_score(test_y, pred_k)))
    print("Accuracy of GBC: {0}".format(accuracy_score(test_y, pred_g)))
    # cross validation
    models = [reg, knn, gbc]
    name_models = ["Logistic", "KNN", "GBC"]
    for j in range(len(models)):
        # setup cross validation
        model = models[j]
        fold_k = 5
        # parameter setting for each model
        if j == 0:  # Logistic Regression
            param_dist = dict(C=stats.uniform(loc=0, scale=4), penalty=['l2', 'l1'])
        elif j == 1:  # KNN
            param_dist = {'n_neighbors': range(3, 21),
                          'metric': ['euclidean', 'manhattan']}
        elif j == 2:  # GBC
            param_dist = {'n_estimators': range(60, 181),
                          'max_depth': [5, 10, 15, 20, 25],
                          'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                          'criterion': ['friedman_mse', 'mse']}
        rand_search = RandomizedSearchCV(model, param_distributions=param_dist, scoring='accuracy',
                                         return_train_score=True, n_jobs=-1)
        # cross validation operation (K-fold and RandomizedSearch)
        scores_kfold = cross_val_score(model, features_sampled[i], target_sampled[i], cv=fold_k)
        rand_search.fit(x_temp, y_temp)
        acc_kfold = np.mean(scores_kfold)
        pred_rs = rand_search.predict(test_x)
        acc_rs = accuracy_score(test_y, pred_rs)
        # show accuracy (cv version)
        print("Accuracy({0} & {1}): {2}".format(name_models[j], "K-Fold", acc_kfold))
        print("Accuracy({0} & {1}): {2}".format(name_models[j], "RS", acc_rs))
        print("Best Parameter of {0}: {1}".format(name_models[j], rand_search.best_params_))
