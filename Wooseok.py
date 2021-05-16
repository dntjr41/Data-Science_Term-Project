########################################################
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
print("\nRobust Scaling")
print(data_rb.head())

# Standard Scaling
StandardScaler = preprocessing.StandardScaler()
data_ss = StandardScaler.fit_transform(data)
data_ss = pd.DataFrame(data_ss)
print("\nStandard Scaling")
print(data_ss.head())

# MinMax Scaling
MinMaxScaler = preprocessing.MinMaxScaler()
data_mm = MinMaxScaler.fit_transform(data)
data_mm = pd.DataFrame(data_mm)
print("\nMinMax Scaling")
print(data_mm.head())

# MaxAbs Scaling
MaxAbsScaler = preprocessing.MaxAbsScaler()
data_ma = MaxAbsScaler.fit_transform(data)
data_ma = pd.DataFrame(data_ma)
print("\nMaxAbs Scaling")
print(data_rb.head())