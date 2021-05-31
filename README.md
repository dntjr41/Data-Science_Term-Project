# Open Source Software


# find_outliers()
 : Use the z-score to handle outlier over mean +- 3SD.
 
 find_outlilers(col)
- parameter 
  ->	col : target dataframe’s column to find outliers
- output 
  ->	index : column corresponding to outlier



# encode_scaling()
 : Encoding categorical data using OriginaEncoder and scaling using RobustScaler.
 
 encode_scaling(df, categories)
- parameter
  -> df : target data to encode and scale
  -> categories : column names of categorical values
- output
  -> data_result : encoded, scaled data



# generate_input_data()
 : Set up the required number of random input data to predict.
 
 generate_input_data(num)
- parameter
  -> num : number of data to create
- output
  -> input_result : generated random data



# generate_valid_data()
 : read validation dataset and set up validation data from kaggle by encoding, scaling 
   and concatenating feature and target data.
 
 generate_valid_data()
- parameter 
  -> (none)
- output 
  -> valid_data : generated validation data



# stroke_prediction()
 : encoding and scaling input dataset using encode_scaled function.
   extract best parameter using GradientBoostingClassifier class and predict operation.
 
 stroke_prediction(df_input)
 - parameter
  -> df_input : data to predict
 - output
  -> pred_final : data finished predicting a model



