# Import Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder
import category_encoders as ce
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from imblearn.over_sampling import SMOTE


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

data2 = data.copy()
target = data["stroke"].copy()
data = data.iloc[:, 0:-1]

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

feature_name = ['id', 'gender', 'age', 'hypertension',
                'heart_disease', 'ever_married', 'work_type',
                'Residence_type', 'avg_glucose_level',
                'bmi', 'smoking_status']

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

# feature selection
print("\n\nfeature selection\n")

# kbest algorithm
print("Kbest algorithm")

# ordinal + robust
k_best = SelectKBest(score_func=f_classif, k = 11)
kb_fit = k_best.fit(data_rb, target)
feature_score = pd.DataFrame(pd.concat([pd.DataFrame(data_rb.columns), pd.DataFrame(kb_fit.scores_)], axis=1))
feature_score.columns = ['feature', 'score']
print(feature_score.nlargest(11, 'score'))
print("\n")

# ordinal + standard
kb_fit = k_best.fit(data_ss, target)
feature_score = pd.DataFrame(pd.concat([pd.DataFrame(data_ss.columns), pd.DataFrame(kb_fit.scores_)], axis=1))
feature_score.columns = ['feature', 'score']
print(feature_score.nlargest(11, 'score'))
print("\n")


# ordinal + minmax
kb_fit = k_best.fit(data_mm, target)
feature_score = pd.DataFrame(pd.concat([pd.DataFrame(data_mm.columns), pd.DataFrame(kb_fit.scores_)], axis=1))
feature_score.columns = ['feature', 'score']
print(feature_score.nlargest(11, 'score'))
print("\n")


# ordinal + maxabs
kb_fit = k_best.fit(data_ma, target)
feature_score = pd.DataFrame(pd.concat([pd.DataFrame(data_ma.columns), pd.DataFrame(kb_fit.scores_)], axis=1))
feature_score.columns = ['feature', 'score']
print(feature_score.nlargest(11, 'score'))
print("\n")


#extra tree algorithm
print("\n\nExtra tree algorithm\n")

# ordinal + robust
extra_tree = ExtraTreesClassifier()
extra_tree.fit(data_rb, target)
feature_imp = pd.Series(extra_tree.feature_importances_, index=data_rb.columns)
feature_imp.nlargest(11).plot(kind='barh')
#plt.show()


# ordinal + standard
extra_tree = ExtraTreesClassifier()
extra_tree.fit(data_ss, target)
feature_imp = pd.Series(extra_tree.feature_importances_, index=data_ss.columns)
feature_imp.nlargest(11).plot(kind='barh')
#plt.show()


# ordinal + minmax
extra_tree = ExtraTreesClassifier()
extra_tree.fit(data_mm, target)
feature_imp = pd.Series(extra_tree.feature_importances_, index=data_mm.columns)
feature_imp.nlargest(11).plot(kind='barh')
#plt.show()


# ordinal + maxabs
extra_tree = ExtraTreesClassifier()
extra_tree.fit(data_ma, target)
feature_imp = pd.Series(extra_tree.feature_importances_, index=data_ma.columns)
feature_imp.nlargest(11).plot(kind='barh')
#plt.show()


#heatmap
m_data = data2.drop(['id'],axis=1)
print(m_data)
corr_matrix = m_data.corr()
print(corr_matrix)
heatmap = sns.heatmap(data2[corr_matrix.index].corr(), annot=True)
#plt.show()



data_rb = data_rb.drop(['gender', 'id', 'Residence_type', 'work_type'], axis=1)
data_ss = data_ss.drop(['gender', 'id', 'Residence_type', 'work_type'], axis=1)
data_mm = data_mm.drop(['gender', 'id', 'Residence_type', 'work_type'], axis=1)
data_ma = data_ma.drop(['gender', 'id', 'Residence_type', 'work_type'], axis=1)


# split train and test data
rb_x_train, rb_x_test, rb_y_train, rb_y_test = train_test_split(data_rb, target, test_size=0.2, shuffle=True,
                                                    stratify=target, random_state=123)
ss_x_train, ss_x_test, ss_y_train, ss_y_test = train_test_split(data_ss, target, test_size=0.2, shuffle=True,
                                                    stratify=target, random_state=123)
mm_x_train, mm_x_test, mm_y_train, mm_y_test = train_test_split(data_mm, target, test_size=0.2, shuffle=True,
                                                    stratify=target, random_state=123)
ma_x_train, ma_x_test, ma_y_train, ma_y_test = train_test_split(data_ma, target, test_size=0.2, shuffle=True,
                                                    stratify=target, random_state=123)


log = LogisticRegression(solver='saga')
knn = KNeighborsClassifier(n_neighbors=7)
gbc = GradientBoostingClassifier(learning_rate=0.8, n_estimators=7)


# Robust scaling data
print("<<<<<<<<<<<<<Robust scaled data fitting>>>>>>>>>>>>>>\n")

# fitting
log.fit(rb_x_train, rb_y_train)
knn.fit(rb_x_train, rb_y_train)
gbc.fit(rb_x_train, rb_y_train)

# predict
log_predict = log.predict(rb_x_test)
knn_predict = knn.predict(rb_x_test)
gbc_predict = gbc.predict(rb_x_test)

# accuracy
print("Accuracy of Logistic: " ,accuracy_score(rb_y_test, log_predict))
print("Accuracy of KNN: ", accuracy_score(rb_y_test, knn_predict))
print("Accuracy of GBC: ", accuracy_score(rb_y_test, gbc_predict))
print("\n")

#cross validation
#log
dist_log = dict(C=stats.uniform(loc=0, scale=4), penalty=['l2', 'l1'])
rand_search_log = RandomizedSearchCV(log, param_distributions=dist_log, scoring='accuracy',
                                 return_train_score=True, n_jobs=-1)

kfold_score = cross_val_score(log, data_rb, target, cv=7)
kfold_accuracy = np.mean(kfold_score)

rand_search_log.fit(rb_x_train, rb_y_train)
rs_predict = rand_search_log.predict(rb_x_test)
rs_accuracy = accuracy_score(rb_y_test, rs_predict)

# log accuracy
print("After cross validation\n")
print("Accuracy(Logistic & K-Fold) : ", kfold_accuracy)
print("Accuracy(Logistic & Rand Search) : ", rs_accuracy)
print("Best Parameter of Logistic : ", rand_search_log.best_params_)

#knn
dist_knn = {'n_neighbors': range(3, 21), 'metric': ['euclidean', 'manhattan']}
rand_search_knn = RandomizedSearchCV(knn, param_distributions=dist_knn, scoring='accuracy',
                                 return_train_score=True, n_jobs=-1)

kfold_score = cross_val_score(knn, data_rb, target, cv=7)
kfold_accuracy = np.mean(kfold_score)

rand_search_knn.fit(rb_x_train, rb_y_train)
rs_predict = rand_search_knn.predict(rb_x_test)
rs_accuracy = accuracy_score(rb_y_test, rs_predict)

# knn accuracy
print("\nAccuracy(KNN & K-Fold) : ", kfold_accuracy)
print("Accuracy(KNN & Rand Search) : ", rs_accuracy)
print("Best Parameter of KNN : ", rand_search_knn.best_params_)

#gbc
dist_gbc = {'n_estimators': range(60, 181),
              'max_depth': [5, 10, 15, 20, 25],
              'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
              'criterion': ['friedman_mse', 'mse']}
rand_search_gbc = RandomizedSearchCV(gbc, param_distributions=dist_gbc, scoring='accuracy',
                                 return_train_score=True, n_jobs=-1)


kfold_score = cross_val_score(gbc, data_rb, target, cv=7)
kfold_accuracy = np.mean(kfold_score)

rand_search_gbc.fit(rb_x_train, rb_y_train)
rs_predict = rand_search_gbc.predict(rb_x_test)
rs_accuracy = accuracy_score(rb_y_test, rs_predict)

# gbc accuracy
print("\nAccuracy(GBC & K-Fold) : ", kfold_accuracy)
print("Accuracy(GBC & Rand Search) : ", rs_accuracy)
print("Best Parameter of GBC : ", rand_search_gbc.best_params_)


###################################################################

# Standard scaling data
print("\n\n<<<<<<<<<<<<Standard scaled data fitting>>>>>>>>>>>>\n")

# fitting
log.fit(ss_x_train, ss_y_train)
knn.fit(ss_x_train, ss_y_train)
gbc.fit(ss_x_train, ss_y_train)

# predict
log_predict = log.predict(ss_x_test)
knn_predict = knn.predict(ss_x_test)
gbc_predict = gbc.predict(ss_x_test)

# accuracy
print("Accuracy of Logistic: " ,accuracy_score(ss_y_test, log_predict))
print("Accuracy of KNN: ", accuracy_score(ss_y_test, knn_predict))
print("Accuracy of GBC: ", accuracy_score(ss_y_test, gbc_predict))
print("\n")

#cross validation
#log
kfold_score = cross_val_score(log, data_ss, target, cv=7)
kfold_accuracy = np.mean(kfold_score)

rand_search_log.fit(ss_x_train, ss_y_train)
rs_predict = rand_search_log.predict(ss_x_test)
rs_accuracy = accuracy_score(ss_y_test, rs_predict)

# log accuracy
print("After cross validation\n")
print("Accuracy(Logistic & K-Fold) : ", kfold_accuracy)
print("Accuracy(Logistic & Rand Search) : ", rs_accuracy)
print("Best Parameter of Logistic : ", rand_search_log.best_params_)

#knn
kfold_score = cross_val_score(knn, data_ss, target, cv=7)
kfold_accuracy = np.mean(kfold_score)

rand_search_knn.fit(ss_x_train, ss_y_train)
rs_predict = rand_search_knn.predict(ss_x_test)
rs_accuracy = accuracy_score(ss_y_test, rs_predict)

# knn accuracy
print("\nAccuracy(KNN & K-Fold) : ", kfold_accuracy)
print("Accuracy(KNN & Rand Search) : ", rs_accuracy)
print("Best Parameter of KNN : ", rand_search_knn.best_params_)

#gbc
kfold_score = cross_val_score(gbc, data_ss, target, cv=7)
kfold_accuracy = np.mean(kfold_score)

rand_search_gbc.fit(ss_x_train, ss_y_train)
rs_predict = rand_search_gbc.predict(ss_x_test)
rs_accuracy = accuracy_score(ss_y_test, rs_predict)

# gbc accuracy
print("\nAccuracy(GBC & K-Fold) : ", kfold_accuracy)
print("Accuracy(GBC & Rand Search) : ", rs_accuracy)
print("Best Parameter of GBC : ", rand_search_gbc.best_params_)


###################################################################

# MinMax scaling data
print("\n\n<<<<<<<<<<<<<<MinMax scaled data fitting>>>>>>>>>>>>>")

# fitting
log.fit(mm_x_train, mm_y_train)
knn.fit(mm_x_train, mm_y_train)
gbc.fit(mm_x_train, mm_y_train)

# predict
log_predict = log.predict(mm_x_test)
knn_predict = knn.predict(mm_x_test)
gbc_predict = gbc.predict(mm_x_test)

# accuracy
print("Accuracy of Logistic: " ,accuracy_score(mm_y_test, log_predict))
print("Accuracy of KNN: ", accuracy_score(mm_y_test, knn_predict))
print("Accuracy of GBC: ", accuracy_score(mm_y_test, gbc_predict))
print("\n")

#cross validation
#log
kfold_score = cross_val_score(log, data_mm, target, cv=7)
kfold_accuracy = np.mean(kfold_score)

rand_search_log.fit(mm_x_train, mm_y_train)
rs_predict = rand_search_log.predict(mm_x_test)
rs_accuracy = accuracy_score(mm_y_test, rs_predict)

# log accuracy
print("After cross validation\n")
print("Accuracy(Logistic & K-Fold) : ", kfold_accuracy)
print("Accuracy(Logistic & Rand Search) : ", rs_accuracy)
print("Best Parameter of Logistic : ", rand_search_log.best_params_)

#knn
kfold_score = cross_val_score(knn, data_mm, target, cv=7)
kfold_accuracy = np.mean(kfold_score)

rand_search_knn.fit(mm_x_train, mm_y_train)
rs_predict = rand_search_knn.predict(mm_x_test)
rs_accuracy = accuracy_score(mm_y_test, rs_predict)

# knn accuracy
print("\nAccuracy(KNN & K-Fold) : ", kfold_accuracy)
print("Accuracy(KNN & Rand Search) : ", rs_accuracy)
print("Best Parameter of KNN : ", rand_search_knn.best_params_)

#gbc
kfold_score = cross_val_score(gbc, data_mm, target, cv=7)
kfold_accuracy = np.mean(kfold_score)

rand_search_gbc.fit(mm_x_train, mm_y_train)
rs_predict = rand_search_gbc.predict(mm_x_test)
rs_accuracy = accuracy_score(mm_y_test, rs_predict)

# gbc accuracy
print("\nAccuracy(GBC & K-Fold) : ", kfold_accuracy)
print("Accuracy(GBC & Rand Search) : ", rs_accuracy)
print("Best Parameter of GBC : ", rand_search_gbc.best_params_)


###################################################################

# MaxAbs scaling data
print("\n\n<<<<<<<<<<<<<<<MaxAbs scaled data fitting>>>>>>>>>>>>>>>>\n")

# fitting
log.fit(ma_x_train, ma_y_train)
knn.fit(ma_x_train, ma_y_train)
gbc.fit(ma_x_train, ma_y_train)

# predict
log_predict = log.predict(ma_x_test)
knn_predict = knn.predict(ma_x_test)
gbc_predict = gbc.predict(ma_x_test)

# accuracy
print("Accuracy of Logistic: " ,accuracy_score(ma_y_test, log_predict))
print("Accuracy of KNN: ", accuracy_score(ma_y_test, knn_predict))
print("Accuracy of GBC: ", accuracy_score(ma_y_test, gbc_predict))
print("\n")

#cross validation
#log
kfold_score = cross_val_score(log, data_ma, target, cv=7)
kfold_accuracy = np.mean(kfold_score)

rand_search_log.fit(ma_x_train, ma_y_train)
rs_predict = rand_search_log.predict(ma_x_test)
rs_accuracy = accuracy_score(ma_y_test, rs_predict)

# log accuracy
print("After cross validation\n")
print("Accuracy(Logistic & K-Fold) : ", kfold_accuracy)
print("Accuracy(Logistic & Rand Search) : ", rs_accuracy)
print("Best Parameter of Logistic : ", rand_search_log.best_params_)

#knn
kfold_score = cross_val_score(knn, data_ma, target, cv=7)
kfold_accuracy = np.mean(kfold_score)

rand_search_knn.fit(ma_x_train, ma_y_train)
rs_predict = rand_search_knn.predict(ma_x_test)
rs_accuracy = accuracy_score(ma_y_test, rs_predict)

# knn accuracy
print("\nAccuracy(KNN & K-Fold) : ", kfold_accuracy)
print("Accuracy(KNN & Rand Search) : ", rs_accuracy)
print("Best Parameter of KNN : ", rand_search_knn.best_params_)

#gbc
kfold_score = cross_val_score(gbc, data_ma, target, cv=7)
kfold_accuracy = np.mean(kfold_score)

rand_search_gbc.fit(ma_x_train, ma_y_train)
rs_predict = rand_search_gbc.predict(ma_x_test)
rs_accuracy = accuracy_score(ma_y_test, rs_predict)

# gbc accuracy
print("\nAccuracy(GBC & K-Fold) : ", kfold_accuracy)
print("Accuracy(GBC & Rand Search) : ", rs_accuracy)
print("Best Parameter of GBC : ", rand_search_gbc.best_params_)
