import os
import uvicorn
from fastapi import FastAPI, HTTPException
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from CustomisedFunctions import bucketize_numerical_inputs,check_input_validity,inputs
from CustomisedFunctions import Main_Numerical_Data,Main_Boolean_Data,Main_Categorical_Data
from TrainModel import TabNetClassifier_1,TabNetClassifier_2


# folder="app/" is for the docker file, folder="" if you are not deploying the api in a docker container
folder="app/"
XGBoost_model_1 = pickle.load(open(os.path.join(os.getcwd(),f'{folder}models/XGBoost_model_1.pkl'),'rb'))
LGB_model_1 = pickle.load(open(os.path.join(os.getcwd(),f'{folder}models/LGB_model_1.pkl'),'rb'))
TabNetClassifier_1.load_weights(f'{folder}models/TabNet_1.ckpt')
XGBoost_model_2 = pickle.load(open(os.path.join(os.getcwd(),f'{folder}models/XGBoost_model_2.pkl'),'rb'))
LGB_model_2 = pickle.load(open(os.path.join(os.getcwd(),f'{folder}models/LGB_model_2.pkl'),'rb'))
TabNetClassifier_2.load_weights(f'{folder}models/TabNet_2.ckpt')
meta_model = pickle.load(open(os.path.join(os.getcwd(),f'{folder}models/meta_model.pkl'),'rb'))
feature_encoder = pickle.load(open(os.path.join(os.getcwd(),f'{folder}models/feature_encoder.pkl'),'rb'))
sc = pickle.load(open(os.path.join(os.getcwd(),f'{folder}models/sc.pkl'),'rb'))

lr = tf.keras.optimizers.schedules.ExponentialDecay(0.0005, decay_steps=100, decay_rate=0.9, staircase=False)
optimizer = tf.keras.optimizers.Adam(lr)
TabNetClassifier_1.compile(optimizer, loss='categorical_crossentropy')
TabNetClassifier_2.compile(optimizer, loss='categorical_crossentropy')

app = FastAPI()

@app.get('/')
def index():
    return {'message': 'hi'}


@app.post('/predict')
def predict_default(data:inputs):
    Numerical_Data=Main_Numerical_Data.copy()
    Boolean_Data=Main_Boolean_Data.copy()
    Categorical_Data=Main_Categorical_Data.copy()
    data = data.dict()
    input_df=check_input_validity(data)

    if type(input_df)==str:
        raise HTTPException(status_code=400, detail=f"ERROR!!!: {input_df}")

    else:
        input_df=bucketize_numerical_inputs(input_df)
        Categorical_Data+=['New_MonthlyCharges','New_tenure','New_TotalCharges']
        customer_id=input_df['customerID'].values
        input_df=pd.concat([input_df[Categorical_Data],input_df[Boolean_Data],input_df[Numerical_Data]],axis=1)

        input_BT=np.array(feature_encoder.transform(input_df))
        input_BT=input_BT.astype('float32')
        sc_col=-3
        input_BT[:,sc_col:]=sc.transform(input_BT[:,sc_col:])
        input_TN=tf.data.Dataset.from_tensor_slices(({i:input_df[i] for i in input_df.columns}))

        BATCH_SIZE=len(input_df)
        xbg_pred_prob_1=XGBoost_model_1.predict_proba(input_BT)
        lgbm_pred_prob_1=LGB_model_1.predict_proba(input_BT)
        Tabnet_pred_prob_1=TabNetClassifier_1.predict(input_TN.batch(BATCH_SIZE))
        xbg_pred_prob_2=XGBoost_model_2.predict_proba(input_BT)
        lgbm_pred_prob_2=LGB_model_2.predict_proba(input_BT)
        Tabnet_pred_prob2=TabNetClassifier_2.predict(input_TN.batch(BATCH_SIZE))

        pred_data=pd.DataFrame()
        pred_data[["xbg_prob_1_0","xbg_prob_1_1"]]=xbg_pred_prob_1
        pred_data[["lgbm_prob_1_0","lgbm_prob_1_1"]]=lgbm_pred_prob_1
        pred_data[["Tabnet_prob_1_0","Tabnet_prob_1_1"]]=Tabnet_pred_prob_1
        pred_data[["xbg_prob_2_0","xbg_prob_2_1"]]=xbg_pred_prob_2
        pred_data[["lgbm_prob_2_0","lgbm_prob_2_1"]]=lgbm_pred_prob_2
        pred_data[["Tabnet_prob_2_0","Tabnet_prob_2_1"]]=Tabnet_pred_prob2

        prediction=meta_model.predict(pred_data.to_numpy())
        prediction=np.argmax(meta_model.predict(pred_data.to_numpy()), 1)
        prediction=np.array(['Yes' if i ==1 else 'No'for i in prediction])
        customer_id=customer_id.reshape(-1,1)
        prediction=prediction.reshape(-1,1)
        prediction=np.concatenate([customer_id,prediction],axis=1)

    return {
        'prediction': prediction
    }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)