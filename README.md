# Open Source Project

We offer best-accuracy model creation operation.  You can make best model for your dataset using this code. This code is open source.


# Source Progress

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
	    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, shuffle=True,  stratify=target, random_state=len(features))  
  
    # setup classifier models  
	reg = LogisticRegression(solver='saga')  
    knn = KNeighborsClassifier(n_neighbors=len(features))  
    gbc = GradientBoostingClassifier(learning_rate=0.8, n_estimators=len(features))  
  
    # fit operation  
	reg.fit(x_train, y_train)  
    knn.fit(x_train, y_train)  
    gbc.fit(x_train, y_train)

Next, using RandomizedSearchCV, do parameter tuning each algorithm like below:

Logistic Regression

    param_dist = dict(C=stats.uniform(loc=l_a, scale=l_b), penalty=l_c)  
	rand_search = RandomizedSearchCV(reg, param_distributions=param_dist, scoring='accuracy',  
	  return_train_score=True, n_jobs=-1)  
	  
	for i in logistic_l:  
	    scores_kfold = cross_val_score(reg, i, target, cv=len(features))  
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
	  
	for i in knn_l:  
	    scores_kfold = cross_val_score(knn, i, target, cv=len(features))  
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
	  
	for i in gbc_l:  
	    scores_kfold = cross_val_score(gbc, i, target, cv=len(features))  
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
