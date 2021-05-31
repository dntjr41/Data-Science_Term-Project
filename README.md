# Open Source Software

# find_outliers()
 : Use the z-score to handle outlier over mean +- 3SD
 find_outlilers(col)
- parameter 
  ->	col : dataframe’s column
- output 
  ->	index : column corresponding to outlier
- code
    def find_outliers(col):
        z = np.abs(stats.zscore(col))
        idx_outliers = np.where(z > 3, True, False)
        return pd.Series(idx_outliers, index=col.index)



# encode_scaling()
 : Encoding categorical data using OriginaEncoder and scaling using RobustScaler
 encode_scaling(df, categories)
- parameter
  -> df : target data, 
  -> categories : column names of categorical variables
- output
  -> data_result : encoded, scaled data
- code 
    def encode_scaling(df, categories):
        for feature in categories:
            encoder = OrdinalEncoder()
            df[feature] = encoder.fit_transform(df[[feature]])
            print(df.head())
        scaler = RobustScaler()
        data_result = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        return data_result



# generate_input_data()
 : Set up the required number of random input data to predict
 generate_input_data(num)
- parameter
  -> num : number of data to create
- output
  -> input_result : generated random data
- code
    def generate_input_data(num):
        random_data = list()
        age_seed = range(1, 81)
        hypertension_seed = [0, 1]
        heart_disease_seed = [0, 1]
        ever_married_seed = ['No', 'Yes']
        for i in range(num):
            temp = list()
            temp.append(random.choice(age_seed))
            temp.append(random.choice(hypertension_seed))
            temp.append(random.choice(heart_disease_seed))
            temp.append(random.choice(ever_married_seed))
            temp.append(round(random.uniform(30, 210), 2))
            random_data.append(temp)
        input_result = pd.DataFrame(random_data, columns=feature_selected_col)
        return input_result



# generate_valid_data()
 : read validation dataset and set up validation data from kaggle by encoding, scaling 
   and concatenating feature and target data
 generate_valid_data()
- parameter 
  -> (none)
- output 
  -> valid_data : generated validation data
- code
    def generate_valid_data():
        valid_raw = pd.read_csv("healthcare-dataset-stroke-data.csv")
        valid_feature = valid_raw[feature_selected_col]
        valid_target = valid_raw[target_col]
        valid_feature = encode_scaling(valid_feature, ['ever_married'])
        valid_data = pd.concat([valid_feature, valid_target], axis=1)
        return valid_data



# stroke_prediction()
 : encoding and scaling input dataset using encode_scaled function
   extract best parameter using GradientBoostingClassifier class
   and predict operation
 stroke_prediction(df_input)
 - parameter
  -> df_input : data to predict
 - output
  -> pred_final : data finished predicting a model
 - code
    def stroke_prediction(df_input):
        input_scaled = encode_scaling(df_input, ['ever_married'])
        best_parameter = rand_cv.best_params_
        model = GradientBoostingClassifier(n_estimators=best_parameter['n_estimators'],
                                           max_depth=best_parameter['max_depth'],
                                           max_features=best_parameter['max_features'],
                                           learning_rate=best_parameter['learning_rate'],
                                           criterion=best_parameter['criterion'],
                                           random_state=37)
        model.fit(x_train, y_train.values.ravel())
        pred_final = model.predict(input_scaled)
        print("The Accuracy of final model: {0}".format(model.score(valid_set[feature_selected_col],
                                                                valid_set[target_col])))
        return pred_final



