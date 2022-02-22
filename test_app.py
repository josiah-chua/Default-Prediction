from fastapi.testclient import TestClient
import pandas as pd

from app import app

client = TestClient(app)

test_df=pd.read_csv('pytest_dataset.csv')

case_1 = test_df.iloc[:5,:].to_dict(orient='list') #proper data
case_2 = test_df.iloc[5:10,:].to_dict(orient='list') #missing data
case_3 = test_df.iloc[10:15,:].to_dict(orient='list') #unacceptable value data
case_4 = test_df.iloc[15:20,:].to_dict(orient='list') #words in numerical column data
case_5 = test_df.iloc[:5,:-3].to_dict(orient='list') #missing feature columns data

print(case_5)

def test_greetings():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "hi"}

def test_prediction():
    response = client.post("/predict",
        json=case_1
    )
    #test if proper input is accepted
    assert response.status_code == 200
    #test if returned prediction is a list of predictions
    assert type(response.json()['prediction'])
    assert all([i in ["Yes","No"]  for i in response.json()['prediction'].values()])

def test_case_2():
    response = client.post("/predict",
        json=case_2
    )
    #test if data sets with missing value will be detected and error will be raised
    assert response.status_code == 400
    assert "ERROR" in response.json()['detail']

def test_case_3():
    response = client.post("/predict",
        json=case_3
    )
     #test if data sets with unacceptable value for boolean features like "maybe" will be detected and error will be raised
    assert response.status_code == 400
    assert "ERROR" in response.json()['detail']

def test_case_4():
    response = client.post("/predict",
        json=case_4
    )
     #test if data sets with words in numerical columns be detected and error will be raised
    assert response.status_code == 400
    assert "ERROR" in response.json()['detail']

def test_case_5():
    response = client.post("/predict",
        json=case_5
    )
     #test if data sets with missing feature columns will be detected and error will be raised
    assert response.status_code == 422
