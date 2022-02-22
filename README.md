# Default-Prediction
This model is a stacked model of boosted trees and TabNets to predict if a person will default.

## Preprocessing and Feature Engineering

To analyse the data, the data was put into Facets a tool for ML data visualization and here are some of the key takeways that were used for feature enginenering and pre processing

### Skewed Data

![image](https://user-images.githubusercontent.com/81459293/154807825-e230e6a9-d02a-42f8-a1bd-5094d11b835d.png)

The data provided is skewed as defaulting is less common, this could lead to overfitting when trianing the model causing the model to be bad at predicting the minority class. This is detrimental for a case of predicting default as it is more important to be able to detect defaulters than non defaulters as the cost to the company is greater. Hence to balance the training dataset, sampling was used.

Oversampling was prefered and used instead of undersampling as it was able to increase the proportion of the minority class while retaining the information from the majority class.

![image](https://user-images.githubusercontent.com/81459293/154808221-80b21154-39be-4bd6-b26b-729b722534e8.png)

SMOTE was used for oversampling. New minority class data are generated by selecting examples that are close in the feature space, and interpolating a new data point between those 2 points, essentially duplicating the minority class with some noise in the data.

### Non-Linear Numerical Data

#### Total Charges

![image](https://user-images.githubusercontent.com/81459293/154806928-23054f39-95bd-425d-b44e-1f2f5fb1f43f.png)

![image](https://user-images.githubusercontent.com/81459293/154806950-e7776d79-256e-4927-a00b-0b1dc706b2fe.png)

#### Tenure

![image](https://user-images.githubusercontent.com/81459293/154807308-1135d3e9-3db4-44e5-8403-cb1ae2791b78.png)

![image](https://user-images.githubusercontent.com/81459293/154807333-015d370e-540a-4c02-90c2-8d6d89c32934.png)

#### Monthly Charges

![image](https://user-images.githubusercontent.com/81459293/154807432-6c9e6a50-6431-4a79-8706-9b92fe409a9f.png)

![image](https://user-images.githubusercontent.com/81459293/154807453-bf7532fd-634b-4171-bfe5-5e05f119d52d.png)

As seen in the graphs for the different numerical features while tenure is slightly linear, the Mothly Charges and Total Charges are not, simply using numerical data might affect the weights for that features as it contributes to the model differently in different ranges. Hence bucketing was done where new catergorical features were engineered based on different effects that different ranges have on the percetage of people who defaulted.

## Model Selection

For Tabular Data most the popular choice would be to use Boosted Trees, as it is efficient at making decision rules from informative feature data. It is considered extremely fast, stable, faster to tune the hyperparameters which is well suited for tabular data. Hence Boosted Trees often out beat Deep learning models in data competitons.

![image](https://user-images.githubusercontent.com/81459293/154810583-8e05aa96-d2ad-4e23-9415-ecead4cf1906.png)

However, newer deep learning models such as Tabnet that have been shown to out perform boosted tress in tabular data predictions. Tabnet uses an attentive layer for features selection essentially allowing for decision boundaries, similar to boosted trees enabling interpretability and better learning as the learning capacity is used for the most useful features. Furthermore the dense layers of the feature transofmrer block also allows for futher feature generation which could allow the model to pick out new features and increase the robustness of the model.

![image](https://user-images.githubusercontent.com/81459293/154811054-b9eb7a04-a89d-4c0f-868e-4ea197c5280d.png)

Hence to maximise the cpabilites of both models, a stacked ensamble of gradient boosted trees, XGBoost and LightGBM, and TabNet which have been shown (paper in refrences) to out perform both models. 2 models of each type was used with the top 2 hyperparmeters for each model which have has cosiderably different structures being used in hopes that the model will be more robust and can predict a greater range of data better. The prediction outputs of the individual different models are then used as inputs for a meta model, in this model a linear regression is used, to have a final prediction.

## Evaluation metric

For skewed data, accuracy is not appropriate to use accuracy to evaluate the model as simply predicting the majority class will result in a good accuracy. Furthermore when we are especially concerned witht he minority class for defaults, it it more suitable to measure how good a model is at correctly predicting that a defaulter will default (recall), but at the same time keeping false positives low (precision). Hence the F1 Score was used as the evaluation metric.

![image](https://user-images.githubusercontent.com/81459293/154809403-dbdee3e1-c270-4320-a21f-d5df79118fa9.png)

As we can see F1 scores work better with imbalance data as it gives equal weight to Precision and Recall, hence the model cannot increase predictions of true positives at the expense of false positives. For further evaulation, it will be used in tendem with precision and recall to see whether the model is performing better after different feature engineering and sampling techniques becasue its is getting better at identifying defaulters (recall increase but precision stays about the same) or if it is because precision is getting better indicating better detection of non defaulters, or if its getting better at identifying defaulters only at the expense of precision.

The F1 metric was also used for hyperparameter turning, where the tops 2 hyperparameters for each model was used in the stacked ensemble

## Performance

![image](https://user-images.githubusercontent.com/81459293/155009338-9ed1d4d1-37e4-4a98-aeab-1bef9d4d7ce9.png)

The evaluation of the performace of the models were tested using 10 fold cross validation and the average F1 score was taken along with the average recall and average precision to see what the change is F1 was due to, a change in recall, a change of precision or both. This cross validation was done with and withoout the sampling and engineered features to see how good were these techniques are at improving performace.

As we can see as a whole the there was an increase in F1 score with more of such techniques employed, with recall increasing with precision decreasing slightly, showing that the models are indeed impoving at its ability to detect defaulters. The odd one out was the TabNet model which worked better with sampling but without the engineered features. this indicates that better feature enginnering techniques could be enmployed or it could be the lack of tuning. 

The stacked ensemble and TabNet had precision decreased significatly when F1 scores increased which has to be looked into more.

Another interesting point is that the stacked ensemble did not do as well as the boosted trees. This could be due to the fact that the individual modeles were tuned seperately, and while boosted trees are fast to tune, Tabnet took a cosiderably longer amount of time to do so. Futher improvement would definiely include turninng the Tabnet model more and doing hyparameter tuning on the stacked model as a whole. Improvements to the meta model could be done to such as putting it though an ANN as the relationship between the models predictions may not be linear and certain preditctions should be allocated more weight.

# How to use

## API

The API was used using FASTAPI and can be run locally.

**Prediction**

POST /predict/{dictionary containing data in json format}

To utilize the model for predictions, a sample.py file is provided. a POST can be made using the python library requests to the local server taking in the data as json format.

The keys for the dictionary containing the data are the feature names and the values are a lists of values in that feature column, allowing for the prediction of multiple predictions

The POST methord will either return a dictionary {'prediction': 2D array of ID and prediction}, or an error code {'details' : ERROR .....} if the input in not formatted correctly.

## Docker

Please ensure that you have docker set up properly on your computer.

### Pull from docker hub 
A Docker image has been put onto Docker Hub and a docker continer with the model can be initalized by pulling it from Docker Hub

![image](https://user-images.githubusercontent.com/81459293/155050181-4c85f88c-2cea-4e6b-b3b5-cecf630a20d7.png)

From the command line use this command to run the docker file, remember to name your container you are running:

![image](https://user-images.githubusercontent.com/81459293/155118837-4ef4de46-b91e-4bde-bab2-cbae047b80d8.png)

Other basic commands

stop:

![image](https://user-images.githubusercontent.com/81459293/155119467-90886312-b07c-4063-8409-202c69ed5c6a.png)

start:

![image](https://user-images.githubusercontent.com/81459293/155119551-f6eefb13-7cdc-4dfc-b684-9e7a63f02590.png)

remove:

![image](https://user-images.githubusercontent.com/81459293/155119977-9c26266a-09a7-4d1b-aced-d77e7515137b.png)


### Create own image from github repository code

Download the zip file from this repository
From the command line enter into the directory and build the docker file with this command, setting anyname to what you want to call the image

![image](https://user-images.githubusercontent.com/81459293/155047806-c2db0769-6171-4a66-8e3b-d8cafd007d05.png)

Create the docker container to deploy the API using this command. Remeber to change the file addresses accroding to the comments as the docker file puts cpies the files into another directory called app/

![image](https://user-images.githubusercontent.com/81459293/155047699-3562dcff-13dd-47af-a7da-f85c258524d2.png)


# Refrences
Facets:https://pair-code.github.io/facets/

TabNet:https://github.com/titu1994/tf-TabNet

Boosted Trees and Deep learning ensamble: https://arxiv.org/pdf/2106.03253.pdf
