# Import Libraries
import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, LabelEncoder, OneHotEncoder, MaxAbsScaler, MinMaxScaler, StandardScaler

# Read the dataset
data = pd.read_csv("train_strokes.csv")

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


categorical_col = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

feature_col = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type',
               'avg_glucose_level', 'bmi', 'smoking_status']

target_col = 'stroke'

target = data[target_col]
features = pd.DataFrame(data.drop(target_col, axis=1), columns=feature_col)



ordinal = OrdinalEncoder()
onehot = OneHotEncoder()
label = LabelEncoder()

standard = StandardScaler()
robust = RobustScaler()
minmax = MinMaxScaler()
maxabs = MaxAbsScaler()

en=[ordinal,onehot]
sc=[standard,robust,minmax]

# parameter
# encoding : selected encoder dataset (ex.[ordinal, onehot, label])
# scaling : selected scaler dataset (ex.[standard, robust, minmax, maxabs])
# features : feature dataset
# target : target dataset
# categories : categorical data columns

# output
# Best Model
def best_accuracy(encoding, scaling, features, target, categories):
    # encoding categorical data
    data_encod = list()

    for encod in encoding:
        data = features.copy()
        for feature in categories:
            if encod == onehot:
                temp = pd.DataFrame(data[feature])
                data[feature] = pd.get_dummies(temp)
            else:
                data[feature] = encod.fit_transform(data[[feature]])
        data_encod.append(data)

    #scaling features
    data_encod_scaled = list()

    for data in data_encod:
        for scale in scaling:
            data_encod_scaled.append(pd.DataFrame(scale.fit_transform(data)))


    target_s = target
    smote = SMOTE(random_state=len(features))
    for i, data in enumerate(data_encod_scaled):
        data_encod_scaled[i], target_s = smote.fit_resample(data, target)

    target = target_s

    print("Input Logistic Regression parameter")
    print("loc(default : -1) : ")
    l_a = int(input())
    print("scale(default : -1) : ")
    l_b = int(input())
    print("penalty(default : -1) : ")
    l_c = input()

    if l_a == -1:
        l_a = 0
    if l_b == -1:
        l_b = 4
    if l_c == '-1':
        l_c = ['l2', 'l1']

    print("\ninput knn parameter")
    print("range (start num) (default : -1) : ")
    k_a = int(input())
    print("range (finish num) (default : -1) : ")
    k_b = int(input())
    print("metric(default : -1) : ")
    k_c = input()

    if k_a == -1:
        k_a = 3
    if k_b == -1:
        k_b = 21
    if k_c == '-1':
        k_c = ['euclidean', 'manhattan']


    print("\ninput gbc parameter")
    print("range(start num) (default : -1) : ")
    g_a = int(input())
    print("range(finish num) (default : -1) : ")
    g_b = int(input())
    print("max_depth(default : -1) : ")
    g_c = input()
    print("learning rate(default : -1) : ")
    g_d = input()
    print("criterion(default : -1) : ")
    g_e = input()

    if g_a == -1:
        g_a = 60
    if g_b == -1:
        g_b = 181
    if g_c == '-1':
        g_c = [5, 10, 15, 20, 25]
    if g_d == '-1':
        g_d = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if g_e == '-1':
        g_e = ['friedman_mse', 'mse']

    best_m = features
    best_a = 0

    for feature in data_encod_scaled:
        x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, shuffle=True, stratify=target, random_state=7)

        # setup classifier models
        reg = LogisticRegression(solver='saga')
        knn = KNeighborsClassifier(n_neighbors = 5)
        gbc = GradientBoostingClassifier(learning_rate = 0.8, n_estimators = 5)

        # fit operation
        reg.fit(x_train, y_train)
        knn.fit(x_train, y_train)
        gbc.fit(x_train, y_train)

        
        # logistic
        param_dist = dict(C=stats.uniform(loc=l_a, scale=l_b), penalty=l_c)
        rand_search = RandomizedSearchCV(reg, param_distributions=param_dist, scoring='accuracy',
                                         return_train_score=True, n_jobs=-1)

        scores_kfold = cross_val_score(reg, feature, target, cv=7)
        rand_search.fit(x_train, y_train)
        acc_kfold = np.mean(scores_kfold)
        pred_rs = rand_search.predict(x_test)
        acc_rs = accuracy_score(y_test, pred_rs)

        if acc_rs > best_a:
            best_a = acc_rs
            best_m = pred_rs
        
        
        # knn
        param_dist = {'n_neighbors': range(k_a, k_b), 'metric': k_c}
        rand_search = RandomizedSearchCV(knn, param_distributions=param_dist, scoring='accuracy',
                                         return_train_score=True, n_jobs=-1)

        scores_kfold = cross_val_score(knn, feature, target, cv=7)
        rand_search.fit(x_train, y_train)
        acc_kfold = np.mean(scores_kfold)
        pred_rs = rand_search.predict(x_test)
        acc_rs = accuracy_score(y_test, pred_rs)

        if acc_rs > best_a:
            best_a = acc_rs
            best_m = pred_rs
        
        # gbc
        param_dist = {'n_estimators': range(g_a, g_b),
                      'max_depth': g_c,
                      'learning_rate': g_d,
                      'criterion': g_e}
        rand_search = RandomizedSearchCV(gbc, param_distributions=param_dist, scoring='accuracy',
                                         return_train_score=True, n_jobs=-1)

        scores_kfold = cross_val_score(gbc, feature, target, cv=7)
        rand_search.fit(x_train, y_train)
        acc_kfold = np.mean(scores_kfold)
        pred_rs = rand_search.predict(x_test)
        acc_rs = accuracy_score(y_test, pred_rs)

        if acc_rs > best_a:
            best_a = acc_rs
            best_m = pred_rs
        
    return best_m




best_accuracy(en,sc,features,target,categorical_col)

