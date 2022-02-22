from sklearn.preprocessing import LabelEncoder
from keras import backend as K
import pandas as pd
import numpy as np


Main_Numerical_Data=['tenure','MonthlyCharges','TotalCharges']
Main_Boolean_Data=['SeniorCitizen','Partner','Dependents','PhoneService','PaperlessBilling']
Main_Categorical_Data=['gender','MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod']
Main_Label=["Default"]

columns_order=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges']

Own_Label_Encoders={
    "SeniorCitizen":LabelEncoder().fit([0.0, 1.0]),
    "Partner":LabelEncoder().fit(['No', 'Yes']),
    "Dependents":LabelEncoder().fit(['No', 'Yes']),
    "PhoneService":LabelEncoder().fit(['No', 'Yes']),
    "PaperlessBilling":LabelEncoder().fit(['No', 'Yes']),
    "gender":LabelEncoder().fit(['Female', 'Male']),
    "MultipleLines":LabelEncoder().fit(['No', 'No phone service', 'Yes']),
    "InternetService":LabelEncoder().fit(['DSL', 'Fiber optic', 'No']),
    "OnlineSecurity":LabelEncoder().fit(['No', 'No internet service', 'Yes']),
    "OnlineBackup":LabelEncoder().fit(['No', 'No internet service', 'Yes']),
    "DeviceProtection":LabelEncoder().fit(['No', 'No internet service', 'Yes']),
    "TechSupport":LabelEncoder().fit(['No', 'No internet service', 'Yes']),
    "StreamingTV":LabelEncoder().fit(['No', 'No internet service', 'Yes']),
    "StreamingMovies":LabelEncoder().fit(['No', 'No internet service', 'Yes']),
    "Contract":LabelEncoder().fit(['Month-to-month', 'One year', 'Two year']),
    "PaymentMethod":LabelEncoder().fit(['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check']),
    "Default":LabelEncoder().fit(['No', 'Yes'])
}

Correct_Data={
    "SeniorCitizen":[0.0, 1.0],
    "Partner":['No', 'Yes'],
    "Dependents":['No', 'Yes'],
    "PhoneService":['No', 'Yes'],
    "PaperlessBilling":['No', 'Yes'],
    "gender":['Female', 'Male'],
    "MultipleLines":['No', 'No phone service', 'Yes'],
    "InternetService":['DSL', 'Fiber optic', 'No'],
    "OnlineSecurity":['No', 'No internet service', 'Yes'],
    "OnlineBackup":['No', 'No internet service', 'Yes'],
    "DeviceProtection":['No', 'No internet service', 'Yes'],
    "TechSupport":['No', 'No internet service', 'Yes'],
    "StreamingTV":['No', 'No internet service', 'Yes'],
    "StreamingMovies":['No', 'No internet service', 'Yes'],
    "Contract":['Month-to-month', 'One year', 'Two year'],
    "PaymentMethod":['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'],
}

tenure_ranges=[0,30,50,1e100]
monthly_ranges=[0,10,30,50,80,90,1e100]
total_ranges=[0,100,1000,5000,9000,1e100]

def bucketize_tenure(x):
    ranges_=tenure_ranges
    for i in range(len(ranges_)-1):
        if x<ranges_[i+1] and x>=ranges_[i]:
            return i

def bucketize_monthlycharge(x):
    ranges_=monthly_ranges
    for i in range(len(ranges_)-1):
        if x<ranges_[i+1] and x>=ranges_[i]:
            return i

def bucketize_TotalCharges(x):
    ranges_=total_ranges
    for i in range(len(ranges_)-1):
        if x<ranges_[i+1] and x>=ranges_[i]:
            return i

def convert_numerical_data(x):
    if type(x)!=str:
        return float(x)
    else:
        if x.replace(" ","")=="":
            return np.NaN
        else:
            try:
                new=float(x.replace(" ",""))
                return new
            except:
                return np.NaN

def check_input_validity(dic):
    try:
        input_df=pd.DataFrame.from_dict(dic)
        for col in input_df.columns:
            if col=='SeniorCitizen':
                input_df[col]=input_df[col].apply(lambda x: float(x))
            elif col in Main_Numerical_Data:
                input_df[col]=input_df[col].apply(convert_numerical_data)
            elif col=="customerID"  :
                pass
            else:
                if sorted(input_df[col].unique())!=Correct_Data[col]:
                    for i in sorted(input_df[col].unique()):
                        if i not in Correct_Data[col]:
                            return f"input data {i} for {col} cannot be accepted"

            if input_df[col].isnull().any():
                return f"cannot accept empty values, empty values found in {col}"

        for i in Main_Boolean_Data+Main_Categorical_Data:
            input_df[i]=Own_Label_Encoders[i].transform(input_df[i])

        return input_df

    except Exception as e:
        return e

def bucketize_numerical_inputs(input_df):
    input_df['New_MonthlyCharges']=input_df['MonthlyCharges'].apply(bucketize_monthlycharge)
    input_df['New_tenure']=input_df['tenure'].apply(bucketize_tenure)
    input_df['New_TotalCharges']=input_df['TotalCharges'].apply(bucketize_TotalCharges)
    return input_df

from pydantic import BaseModel

class inputs(BaseModel):
    customerID:list
    gender:list
    SeniorCitizen:list
    Partner:list
    Dependents:list
    tenure:list
    PhoneService:list
    MultipleLines:list
    InternetService:list
    OnlineSecurity:list
    OnlineBackup:list
    DeviceProtection:list
    TechSupport:list
    StreamingTV:list
    StreamingMovies:list
    Contract:list
    PaperlessBilling:list
    PaymentMethod:list
    MonthlyCharges:list
    TotalCharges:list
