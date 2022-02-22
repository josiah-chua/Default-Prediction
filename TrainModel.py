import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import confusion_matrix,recall_score,precision_score, f1_score
from CustomisedFunctions import Main_Numerical_Data,Main_Boolean_Data,Main_Categorical_Data,Main_Label
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import  LinearRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from TabNet import TabNetClassifier
from CustomisedFunctions import Own_Label_Encoders,bucketize_monthlycharge,bucketize_tenure, bucketize_TotalCharges
import pickle

Numerical_Data=Main_Numerical_Data.copy()
Boolean_Data=Main_Boolean_Data.copy()
Categorical_Data=Main_Categorical_Data.copy()
Label=Main_Label.copy()

# folder="app/" is for the docker file, folder="" if you are not deploying the api in a docker container
folder="app/"
df=pd.read_csv(f'{folder}finantier_data_technical_test_dataset.csv')

# Remove usless empty column at the back
df=df.iloc[:,:-1]

# Relabel columns
new_columns=list(df.columns)+['Default']
df=df.rename(columns={new_columns[i]:new_columns[i+1] for i in range(len(new_columns)-1)})
df.index.names = [new_columns[0]]

#convert totalcharges column from string to values and remove any null values
df['TotalCharges']=df['TotalCharges'].apply(lambda x: round(float(x),2) if x!=" " else np.NaN)
df=df.loc[df.isnull().any(axis=1)==False]

# Label encode all Categorical boolean and Label Data (For Tabnet input)
for i in Boolean_Data+Categorical_Data+Label:
  df[i]=Own_Label_Encoders[i].transform(df[i])

# Bucketise numerical features to feature engineer new catergorial data (refer to github discription for explanation)
df['New_MonthlyCharges']=df['MonthlyCharges'].apply(bucketize_monthlycharge)
Categorical_Data+=['New_MonthlyCharges']
df['New_tenure']=df['tenure'].apply(bucketize_tenure)
Categorical_Data+=['New_tenure']
df['New_TotalCharges']=df['TotalCharges'].apply(bucketize_TotalCharges)
Categorical_Data+=['New_TotalCharges']
x=pd.concat([df[Categorical_Data],df[Boolean_Data],df[Numerical_Data]],axis=1)
y=df[Label].values

#reduce test size close to 0 to do a final training on all data
x_train,x_test,y_train, y_test=train_test_split(x,y,test_size=0.1)

#oversample test data to make data less skewed (refer to github discription for explanation)
oversample = SMOTE(sampling_strategy=0.6)
x_train, y_train = oversample.fit_resample(x_train, y_train)

# Manage Boosted Trees' Input

    # Find the cut off between train and test set need to remerge for encoding
cut=len(x_train)
x_BT=pd.concat([x_train,x_test], axis=0)
feature_encoder=ColumnTransformer(transformers=[("OHE",OneHotEncoder(sparse=False),Categorical_Data)],remainder="passthrough")
x_BT=np.array(feature_encoder.fit_transform(x_BT))

    #resplit test and training set
x_BT_train=x_BT[:cut]
x_BT_test=x_BT[cut:]

    # Numerical columns to be normalized
sc_col=-3

sc=StandardScaler()
x_BT_train[:,sc_col:]=sc.fit_transform(x_BT_train[:,sc_col:])
x_BT_test[:,sc_col:]=sc.transform(x_BT_test[:,sc_col:])
y_BT_train=y_train

# Manage TabNet's Input
y_TN_train=y_train
y_TN_train=y_TN_train.reshape(-1,1)

OHE_y=OneHotEncoder(sparse=False)

OHE_y.fit([[0],[1]])
y_TN_train=OHE_y.transform(y_TN_train)
x_TN_train=x_train
x_TN_test=x_test

ds_train=tf.data.Dataset.from_tensor_slices(({i:x_TN_train[i] for i in x_TN_train.columns},y_TN_train))
ds_train_no_labels=tf.data.Dataset.from_tensor_slices(({i:x_TN_train[i] for i in x_TN_train.columns}))
ds_test=tf.data.Dataset.from_tensor_slices(({i:x_TN_test[i] for i in x_TN_test.columns}))

BATCH_SIZE = 100

ds_train = ds_train.batch(BATCH_SIZE)

#feature columns so that even a feature is one hot encoded, columns are identified together a features which will help in the model picking out features 
feature_columns=[]
feature_columns+=[tf.feature_column.numeric_column(i) for i in Numerical_Data]
feature_columns+=[tf.feature_column.bucketized_column(tf.feature_column.numeric_column(i),boundaries=[0.5 + j for j in range(len(df[i].unique())-1)]) for i in Boolean_Data]
feature_columns+=[tf.feature_column.bucketized_column(tf.feature_column.numeric_column(i),boundaries=[0.5 + j for j in range(len(df[i].unique())-1)]) for i in Categorical_Data]


#Train individual models stacked model
XGBoost_model_1 = XGBClassifier(max_depth=4 , learning_rate= 0.036213820332206915, n_estimators=664, 
colsample_bytree=0.09, gamma=2.7600000000000002, min_child_weight=0.2, subsample=0.3, 
reg_lambda=0.55,use_label_encoder=False).fit(x_BT_train,y_BT_train)

XGBoost_model_2 = XGBClassifier(max_depth=14, learning_rate=0.01380228667318763, n_estimators=165, 
colsample_bytree=0.2, gamma=3.0100000000000002, min_child_weight=1.2000000000000002, subsample=0.3, max_bin=500, 
reg_lambda=0.14,use_label_encoder=False).fit(x_BT_train,y_BT_train)

LGB_model_1= LGBMClassifier(n_estimators=209,learning_rate=0.011909390341641951, 
num_leaves=48,max_depth=8,min_child_samples=754,
colsample_bytree=0.5080520180651997,max_bin=840,boosting_type='gbdt').fit(x_BT_train,y_BT_train)

LGB_model_2 = LGBMClassifier(n_estimators=135,learning_rate=0.026640699922665663, 
num_leaves=112,max_depth=12,min_child_samples=395,
colsample_bytree=0.24795028091424715,max_bin=790,boosting_type='gbdt').fit(x_BT_train,y_BT_train)

TabNetClassifier_1=TabNetClassifier(feature_columns, num_features=58, num_classes=2,
feature_dim=197, output_dim=34,num_decision_steps=7, relaxation_factor=1.04,
sparsity_coefficient=1e-5, batch_momentum=0.98,epsilon=1e-5)

TabNetClassifier_2=TabNetClassifier(feature_columns, num_features=58, num_classes=2,
feature_dim=172, output_dim=25,num_decision_steps=4, relaxation_factor=1.045,
sparsity_coefficient=1e-5, batch_momentum=0.98,epsilon=1e-5)

if __name__ == '__main__':

    checkpoint_path_1= "models/TabNet_1.ckpt"
    checkpoint_path_2= "models/TabNet_2.ckpt"

    # Create a callback that saves the model's weights
    cp_callback_1 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_1,save_weights_only=True,verbose=1)
    cp_callback_2 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_2,save_weights_only=True,verbose=1)

    lr_1 = tf.keras.optimizers.schedules.ExponentialDecay(0.0005, decay_steps=100, decay_rate=0.9, staircase=False)
    optimizer_1 = tf.keras.optimizers.Adam(lr_1)
    TabNetClassifier_1.compile(optimizer_1, loss='categorical_crossentropy')
    TabNetClassifier_1.fit(ds_train, epochs=36,verbose=2,callbacks=[cp_callback_1])

    lr_2 = tf.keras.optimizers.schedules.ExponentialDecay(0.0005, decay_steps=100, decay_rate=0.9, staircase=False)
    optimizer_2 = tf.keras.optimizers.Adam(lr_2)
    TabNetClassifier_2.compile(optimizer_2, loss='categorical_crossentropy')
    TabNetClassifier_2.fit(ds_train, epochs=36,verbose=2,callbacks=[cp_callback_2])

    #get the probability of each prediction of training set to be used as inputs for the meta model
    xbg_prob_1=XGBoost_model_1.predict_proba(x_BT_train)
    lgbm_prob_1=LGB_model_1.predict_proba(x_BT_train)
    Tabnet_prob_1=TabNetClassifier_1.predict(ds_train_no_labels.batch(BATCH_SIZE))
    xbg_prob_2=XGBoost_model_2.predict_proba(x_BT_train)
    lgbm_prob_2=LGB_model_2.predict_proba(x_BT_train)
    Tabnet_prob_2=TabNetClassifier_2.predict(ds_train_no_labels.batch(BATCH_SIZE))

    data=pd.DataFrame()
    data[["xbg_prob_1_0","xbg_prob_1_1"]]=xbg_prob_1
    data[["lgbm_prob_1_0","lgbm_prob_1_1"]]=lgbm_prob_1
    data[["Tabnet_prob_1_0","Tabnet_prob_1_1"]]=Tabnet_prob_1
    data[["xbg_prob_2_0","xbg_prob_2_1"]]=xbg_prob_2
    data[["lgbm_prob_2_0","lgbm_prob_2_1"]]=lgbm_prob_2
    data[["Tabnet_prob_2_0","Tabnet_prob_2_1"]]=Tabnet_prob_2

    meta_model=LinearRegression()
    meta_model.fit(data.to_numpy(), y_TN_train)

    #prediciting test set
    xbg_pred_prob_1=XGBoost_model_1.predict_proba(x_BT_test)
    lgbm_pred_prob_1=LGB_model_1.predict_proba(x_BT_test)
    Tabnet_pred_prob_1=TabNetClassifier_1.predict(ds_test.batch(BATCH_SIZE))
    xbg_pred_prob_2=XGBoost_model_2.predict_proba(x_BT_test)
    lgbm_pred_prob_2=LGB_model_2.predict_proba(x_BT_test)
    Tabnet_pred_prob2=TabNetClassifier_2.predict(ds_test.batch(BATCH_SIZE))

    pred_data=pd.DataFrame()
    pred_data[["xbg_prob_1_0","xbg_prob_1_1"]]=xbg_pred_prob_1
    pred_data[["lgbm_prob_1_0","lgbm_prob_1_1"]]=lgbm_pred_prob_1
    pred_data[["Tabnet_prob_1_0","Tabnet_prob_1_1"]]=Tabnet_pred_prob_1
    pred_data[["xbg_prob_2_0","xbg_prob_2_1"]]=xbg_pred_prob_2
    pred_data[["lgbm_prob_2_0","lgbm_prob_2_1"]]=lgbm_pred_prob_2
    pred_data[["Tabnet_prob_2_0","Tabnet_prob_2_1"]]=Tabnet_pred_prob2


    #Evaluation of model
    y_pred=np.argmax(meta_model.predict(pred_data.to_numpy()), 1)

    con_matrix=confusion_matrix(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    f1_score_=f1_score(y_test,y_pred)

    print(f"{con_matrix}")
    print(f"precison_score {precision}")
    print(f"recall {recall}")
    print(f"f1_score_ {f1_score_}")


    #saving models, featuere encoder and Standard Scalar

    filename = 'models/XGBoost_model_1.pkl'
    pickle.dump(XGBoost_model_1, open(filename, 'wb'))
    filename = 'models/LGB_model_1.pkl'
    pickle.dump(LGB_model_1, open(filename, 'wb'))
    filename = 'models/XGBoost_model_2.pkl'
    pickle.dump(XGBoost_model_2, open(filename, 'wb'))
    filename = 'models/LGB_model_2.pkl'
    pickle.dump(LGB_model_2, open(filename, 'wb'))
    filename = 'models/meta_model.pkl'
    pickle.dump(meta_model, open(filename, 'wb'))
    filename = 'models/feature_encoder.pkl'
    pickle.dump(feature_encoder, open(filename, 'wb'))
    filename = 'models/sc.pkl'
    pickle.dump(sc, open(filename, 'wb'))
