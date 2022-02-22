import pandas as pd
import requests as r
import json

# format of the data dictionary in the post method
keys={'customerID':['List of strings'], 
      "gender": ['List of strings (Male or Female)'], 
      "SeniorCitizen": ["List of floats 1.0 for yes 0.0 for no"], 
      "Partner":['List of strings (Yes or No)'], 
      "Dependents":['List of strings (Yes or No)'], 
      "tenure": ["List of floats"],
      "PhoneService":['List of strings (Yes or No)'], 
      "MultipleLines":['List of strings (Yes or No)'], 
      "InternetService":['List of strings (Yes or No)'], 
      "OnlineSecurity":['List of strings (Yes or No)'],
      "OnlineBackup":['List of strings (Yes or No)'], 
      "DeviceProtection":['List of strings (Yes or No)'], 
      "TechSupport":['List of strings (Yes or No)'],
      "StreamingTV":['List of strings (Yes or No)'],
      "StreamingMovies":['List of strings (Yes or No)'], 
      "Contract":['List of strings (Yes or No)'], 
      "PaperlessBilling":['List of strings (Yes or No)'], 
      "PaymentMethod":['List of strings (Yes or No)'],
      "MonthlyCharges": ["List of floats"], 
      "TotalCharges":["List of floats"]
      }

# Get dataset from CSV file
data_df=pd.read_csv('your_data_dataset.csv')

# convert data to dictionary
data_df_dic=data_df.to_dict(orient='list')

#send a POST method
prediction = r.post("http://127.0.0.1:8000/predict", json=data_df_dic)
results = prediction.json()

try:
    #try to print out results
    print(results["prediction"])
except:
    #Visualize error if there are no predictions
    print(results["details"])
