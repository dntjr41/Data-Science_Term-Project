#### Open Source Software

-----------------------------

## best_accuracy()
: Analyze all combinations of scalers and encoders to produce the highest accuracy combination output

best_accuracy(features, target, categories)
- parameter
 -> features : feature dataset
 -> target : target dataset
 -> categories : categorical data columns

- output 
 -> Best Accuracy and encoder, scaler combination of Logistic Regression, KNN, GBC 
 
# code

 def best_accuracy(features, target, categories):
     data_encod = list()

     data = features.copy()
     for feature in categories:
         data[feature] = OrdinalEncoder().fit_transform(data[[feature]])
     data_encod.append(data)

     data = features.copy()
     for feature in categories:
         temp = pd.DataFrame(data[feature])
         data[feature] = pd.get_dummies(temp)
     data_encod.append(data)

     data = features.copy()
     for feature in categories:
         data[feature] = LabelEncoder().fit_transform(data[[feature]])
     data_encod.append(data)


     encod_scale = ['ordinal & minmax','ordinal & standard', 'ordinal & robust', 'ordinal & maxabs'
                    ,'onehot & minmax','onehot & standard', 'onehot & robust', 'onehot & maxabs'
                    ,'label & minmax','label & standard', 'label & robust', 'label & maxabs']

     #scaling features
     data_encod_scaled = list()

     for data in data_encod:
         data_encod_scaled.append(pd.DataFrame(MinMaxScaler().fit_transform(data)))
         data_encod_scaled.append(pd.DataFrame(StandardScaler().fit_transform(data)))
         data_encod_scaled.append(pd.DataFrame(RobustScaler().fit_transform(data)))
         data_encod_scaled.append(pd.DataFrame(MaxAbsScaler().fit_transform(data)))

     logistic_l = list()
     knn_l = list()
     gbc_l = list()

     for feature in data_encod_scaled:
         x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, shuffle=True, stratify=target, random_state=7)

         # setup classifier models
         reg = LogisticRegression(solver='saga')
         knn = KNeighborsClassifier(n_neighbors=5)
         gbc = GradientBoostingClassifier(learning_rate=0.8, n_estimators=5)

         # fit operation
         reg.fit(x_train, y_train)
         knn.fit(x_train, y_train)
         gbc.fit(x_train, y_train)

         # prediction to check accuracy of model
         pred_r = reg.predict(x_test)
         pred_k = knn.predict(x_test)
         pred_g = gbc.predict(x_test)

         logistic_l.append(accuracy_score(y_test,pred_r))
         knn_l.append(accuracy_score(y_test,pred_k))
         gbc_l.append(accuracy_score(y_test,pred_g))

     print(logistic_l)
     print(knn_l)
     print(gbc_l)

     l_highest = 0;
     l_index = 0;
     for i, accuracy in enumerate(logistic_l):
         if l_highest < accuracy:
             l_highest = accuracy;
             l_index = i;

     k_highest = 0;
     k_index = 0;
     for i, accuracy in enumerate(knn_l):
         if k_highest < accuracy:
             k_highest = accuracy;
             k_index = i;

     g_highest = 0;
     g_index = 0;
     for i, accuracy in enumerate(gbc_l):
         if g_highest < accuracy:
             g_highest = accuracy;
             g_index = i;

     print("Best Accuracy of Logistic Regression is ", encod_scale[l_index], " : ", l_highest)
     print("Best Accuracy of KNN is ", encod_scale[k_index], " : ", k_highest)
     print("Best Accuracy of GBC is ", encod_scale[g_index], " : ", g_highest)

     return 0

