print("\n############ Before encoding ############")
print(data.head())
print(data.info())

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

# work_type
# ['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'children'] = [0,1,2,3,4]
work = data[["work_type"]]
work_encod = ordinalencoder.fit_transform(work)
data['work_type']=work_encod

# Residence_type
# ['Rural', 'Urban'] = [0,1]
res = data[["Residence_type"]]
res_encod = ordinalencoder.fit_transform(res)
data['Residence_type'] = res_encod

# smoking_status
# ['formerly smoked', 'never smoked', 'smokes'] = [0,1,2]
smoke = data[["smoking_status"]]
smoke_encod = ordinalencoder.fit_transform(smoke)
data['smoking_status'] = smoke_encod


# float to int
data['gender'] = data['gender'].astype(int)
data['ever_married'] = data['ever_married'].astype(int)
data['work_type'] = data['work_type'].astype(int)
data['Residence_type'] = data['Residence_type'].astype(int)
data['smoking_status']=data['smoking_status'].astype(int)


print("\n\n############ After encoding ############")
print(data.head())
print(data.info())


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
