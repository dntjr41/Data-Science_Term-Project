### Data Science Term Project - Heart Stroke Prediction
***
* Open Source Project

1. Objective Setting
2. Data Curation
3. Data Inspection
4. Data Preprocessing
5. Data Analysis
6. Data Evaluation
7. Conclusion

***
# Open Source Project

We offer best-accuracy model creation operation.  You can make best model for your dataset using this code. This code is open source.


# Source Description

If you want to use this operation, copy or download this source code and call 'best_accuracy()' function. If your dataset is needed another pre-processing, you should do pre-processing before using this function. But this function can only use in classification problem.

## Parameter

best_accuracy() function consists of 5 parameters.

 - **encoding** (*Type = list of Encoder object*): You can set up encoding algorithm in scikit-learrn library in the form of encoder object list.
 -  **scaling** (*Type = list of Scaler object*): You can set up scaling algorithm in scikit-learn library in the form of scaler object list.
 - **features** (*Type = DataFrame object*): This parameter is features part in your raw dataset in the form of DataFrame object in pandas library.
 - **target** (*Type = Series object*): This parameter is target values in your raw dataset in the form of Series object in pandas library.
 - **categories** (*Type = list of String*): This parameter is the index of categorical data columns in your raw dataset. It is necessary to encoding.

## In-Progress

Once all parameters is obtained, encoding and scaling using all method in parameters like below:

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
    
    data_encod_scaled = list()
    for data in data_encod:  
	    for scale in scaling:  
	        data_encod_scaled.append(pd.DataFrame(scale.fit_transform(data)))
        
Data encoding and scaling are successful, do oversampling to solve the data imbalance problem.

    smote = SMOTE(random_state=len(features))  
	for i, data in enumerate(data_encod_scaled):  
	    data_encod_scaled[i], target_s = smote.fit_resample(data, target)

Then, fit 3 algorithm (Logistic Regression, K-Nearest Neighbors, Gradient Boosting Classifier) using pre-processed data.

    for feature in data_encod_scaled:  
	    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, shuffle=True,  stratify=target, random_state=7)  
  
    # setup classifier models  
	reg = LogisticRegression(solver='saga')  
    knn = KNeighborsClassifier(n_neighbors=5))  
    gbc = GradientBoostingClassifier(learning_rate=0.8, n_estimators=5)  
  
    # fit operation  
	reg.fit(x_train, y_train)  
    knn.fit(x_train, y_train)  
    gbc.fit(x_train, y_train)

Next, using RandomizedSearchCV, do parameter tuning each algorithm like below:

Logistic Regression

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

K-Nearest Neighbors

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

Gradient Boosting Classifier

    param_dist = {'n_estimators': range(g_a, g_b),  
	  'max_depth': g_c,  
	  'learning_rate': g_d,  
	  'criterion': g_e}  
	rand_search = RandomizedSearchCV(gbc, param_distributions=param_dist, scoring='accuracy',  
	  return_train_score=True, n_jobs=-1)  
	  
	  
	scores_kfold = cross_val_score(gbc, i, target, cv=7)  
	rand_search.fit(x_train, y_train)  
	acc_kfold = np.mean(scores_kfold)  
	pred_rs = rand_search.predict(x_test)  
	acc_rs = accuracy_score(y_test, pred_rs)  
	  
	if acc_rs > best_a:  
	    best_a = acc_rs  
	    best_m = pred_rs

## Output

Finally, you can get the model that has the best accuracy through the best parameter and method.

    best_accuracy(en, sc, features, target, categorical_col)


# License

 - Department of Software, Gachon University
 - It is **Free Open Source**

***
1. Objective Setting
* This is the era of smart healthcare.
* We set the goal of self-examining for heart disease through large data analysis and some machine learning!
<br>

* ### The Accuracy of Final Model: 0.9211350293532074
* Predict heart stroke – 92.11%
* Assume a million people use the prediction.
* Carotid ultrasound, the simplest heart disease test, costs 100,000 won.
* Cardiac surgery costs 35,000,000 won.
<br>

* Predict heart stroke – 92.11%
* 1,000,000 (people) * 100,000 won (cardiac test costs)
* 9,211 (people) * 35,000,000 won (surgery costs) 
* 422,385,000,000 won (save per year) – 4,223 억

***
2. Data Curation
* [Kaggle - Heart Stroke](https://www.kaggle.com/lirilkumaramal/heart-stroke)

***
3. Data Inspection
* <img width="800" alt="1" src="https://user-images.githubusercontent.com/67234937/121794123-dcd74200-cc40-11eb-8acb-849b76173f1f.PNG">
* <img width="800" alt="2" src="https://user-images.githubusercontent.com/67234937/121794129-e791d700-cc40-11eb-8a62-5cccfec829a8.PNG">
* <img width="800" alt="3" src="https://user-images.githubusercontent.com/67234937/121794131-eb255e00-cc40-11eb-8c7c-7c4eb919a9e8.PNG">
* <img width="800" alt="4" src="https://user-images.githubusercontent.com/67234937/121794134-ef517b80-cc40-11eb-9eea-f80e57a3b92b.PNG">
* <img width="800" alt="5" src="https://user-images.githubusercontent.com/67234937/121794138-f2e50280-cc40-11eb-9f6c-de16093f6079.PNG">
* <img width="800" alt="6" src="https://user-images.githubusercontent.com/67234937/121794139-f7112000-cc40-11eb-91d0-003db15f102f.PNG">
* <img width="800" alt="7" src="https://user-images.githubusercontent.com/67234937/121794141-fa0c1080-cc40-11eb-8244-e8fde682f0d0.PNG">
* <img width="800" alt="8" src="https://user-images.githubusercontent.com/67234937/121794142-fe382e00-cc40-11eb-882a-f8bd028430fd.PNG">
* <img width="800" alt="9" src="https://user-images.githubusercontent.com/67234937/121794144-03957880-cc41-11eb-9fa7-e63bda9c29cb.PNG">

***
4. Data Preprocessing <br>
A. Data Restructuring <br>
B. Data Value Changes <br>
C. Feature Engineering <br>
D. Data Reduction <br>
E. Re-Weight the unbalanced Target Data! <br><br>

* A & B. Data Restructuring & Value Changes
* Cleaning Dirty Data
* Missing data, Unusable Data, Outliers

* Data Encoding
* Label / Ordinal / One – hot Encoding

* Data Scaling
* Standard / Robust / MinMax / MaxAbs Scaling 

* <img width="800" alt="10" src="https://user-images.githubusercontent.com/67234937/121794274-f88f1800-cc41-11eb-94c0-769716b99678.PNG">
* <img width="800" alt="11" src="https://user-images.githubusercontent.com/67234937/121794276-fb8a0880-cc41-11eb-8ec6-d0e7aa8ef556.PNG">
* <img width="800" alt="12" src="https://user-images.githubusercontent.com/67234937/121794277-ff1d8f80-cc41-11eb-817c-39ee674230a5.PNG">
* <img width="800" alt="13" src="https://user-images.githubusercontent.com/67234937/121794280-02b11680-cc42-11eb-81bc-f22b1e659d39.PNG">
* <img width="800" alt="14" src="https://user-images.githubusercontent.com/67234937/121794283-06dd3400-cc42-11eb-9648-062c0d9307ea.PNG">
* <img width="800" alt="15" src="https://user-images.githubusercontent.com/67234937/121794285-0b095180-cc42-11eb-9fea-c4970809ba2f.PNG">
* <img width="800" alt="16" src="https://user-images.githubusercontent.com/67234937/121794291-0f356f00-cc42-11eb-94ec-2035af1cfb04.PNG">
* <img width="800" alt="17" src="https://user-images.githubusercontent.com/67234937/121794300-13618c80-cc42-11eb-8e13-3243b9863552.PNG">

* C. Feature Engineering & D. Data Reduction
* Feature Selection
* Filter Method – ExtraTreeClassifier / HeatMap / KBest

* Data Reduction
* Feature Removal

* Feature Selection & Data Reduction
* KBest + Heatmap + ExtratreeClassifier
* After selecting the 5 features of the 4 scaling in each encoding, The remaining features are dropped

* Select 5 features
* ‘Age’, ‘hypertension’, ‘heart_disease’, ‘ever_married’, ‘avg_glucose_level’

* <img width="800" alt="18" src="https://user-images.githubusercontent.com/67234937/121794368-781ce700-cc42-11eb-8ddc-c84ad6146bfd.PNG">
* <img width="800" alt="19" src="https://user-images.githubusercontent.com/67234937/121794373-7f43f500-cc42-11eb-8839-b4003aa13f6f.PNG">

***
5. Data Analysis
* K – Nearest Neighbors - Knn = KNeighborsClassifier(n_neighbors = k)
* Logistic Regression - Reg = LogisticRegression(solver=‘saga’)
* Gradient Boosting - Gbc = GradientBoostingClassifier(Learning_rate = ratio, n_estimators=e)

* K – Nearest Neighbors <br>
 Parameter = {‘n_neighbors’: range(3, 21), <br>
	        ‘metric’: [‘Euclidean’, ‘manhatten’]}

* Logistic Regression <br>
 Parameter = dict(C=stats.uniform(loc=0, <br>
 		       scale=4), penalty=[‘l2’, ‘l1’])

* Gradient Boosting <br>
 Parameter = {‘n_estimators’: range(100, 301, 50), <br>
	       ‘max_depth’: range(1, 21), <br>
	       ‘max_features’: [‘auto’, ‘sqrt’, ‘log2’], <br>
	       ‘learning_rate’: [0.1, 0.2, 0.3, --- 0.8, 0.9, 1.0], <br>
	       ‘criterion’: [‘friedman_mse’, ‘mse’]}

***
6. Evaluation
* K(5) - fold Cross Validation & Randomized Search
* K – fold = cross_val_score(model, features_sampled[i], <br>
			target_sampled[i], cv = 5) <br>
  Random = RandomizedSearchCV(model, parameter, <br>
	    scoring=‘accuracy’, return_train_score=True, n_jobs=-1)

* <img width="800" alt="20" src="https://user-images.githubusercontent.com/67234937/121794497-9c2cf800-cc43-11eb-9de8-c77ec74217df.PNG">
* <img width="800" alt="21" src="https://user-images.githubusercontent.com/67234937/121794499-a0591580-cc43-11eb-87f6-fe8afb766ac9.PNG">
* <img width="800" alt="22" src="https://user-images.githubusercontent.com/67234937/121794502-a4853300-cc43-11eb-9391-bd2a3e306a61.PNG">
* <img width="800" alt="23" src="https://user-images.githubusercontent.com/67234937/121794504-a818ba00-cc43-11eb-8568-3c4d90a60939.PNG">
* <img width="800" alt="24" src="https://user-images.githubusercontent.com/67234937/121794505-ab13aa80-cc43-11eb-8349-ef4804511442.PNG">

***
7. Conclusion
* <img width="800" alt="25" src="https://user-images.githubusercontent.com/67234937/121794514-c4b4f200-cc43-11eb-958b-cd7c37ccb563.PNG">
* <img width="800" alt="26" src="https://user-images.githubusercontent.com/67234937/121794515-c8e10f80-cc43-11eb-8f71-38d68a4dfd4b.PNG">
* <img width="800" alt="27" src="https://user-images.githubusercontent.com/67234937/121794517-cc749680-cc43-11eb-9b51-77794d62d3a0.PNG">
* <img width="800" alt="28" src="https://user-images.githubusercontent.com/67234937/121794521-cf6f8700-cc43-11eb-9638-13893a50fe16.PNG">
* <img width="800" alt="29" src="https://user-images.githubusercontent.com/67234937/121794524-d39ba480-cc43-11eb-8a79-7bb357b825ab.PNG">

* What we have learned doing the term project
* 심우석 – Until now, when I was doing the homework, I felt that I was lacking in myself <br> 
	   because I had done it by module. However, the term project was able to make <br>
           for the lack of an ‘end-to-end process’ as a whole. And through the process of <br>	
           self-exploring way to solve the problem of data imbalance, I found out that <br>
	   there are various ways to increase the accuracy. <br>

* 나민수 – I thought I would get good results if I followed the lecture, but I didn’t. I was <br>
	   able to learn more by combining various ways myself and trying, and I <br>
	   realized that it was important to communicate and collaborate with team members. <br>

* 남궁다비 – Through this project, I think I was able to repeat what I learned during the <br>
	     semester in the order of data analysis. There was practice other than the project, <br>
             but I was able to enjoy the process of producing practical results through the analysis. <br>

* 201636417 심우석 (1/3) <br>
  Data Preprocessing, Create proposal, final presentation, Source documentation

* 201735824 나민수 (1/3) <br>
  Data Analysis, Evaluation, Implement Main code, Open source documentation

* 201731814 남궁다비 (1/3) <br>
  Data Analysis, Evaluation, Implement Open source
