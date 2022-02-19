# Default-Prediction
This model is a stacked model of boosted trees and TabNets to predict if a person will default.

## Preprocessing and Feature Engineering

To analyse the data, the data was put into Facets a tool for ML data visualization and here are some of the key takeways that were used for feature enginenering and pre processing

### Skewed Data

### Highly Non-Linear Numerical Data

#### Total Charges

![image](https://user-images.githubusercontent.com/81459293/154806928-23054f39-95bd-425d-b44e-1f2f5fb1f43f.png)

![image](https://user-images.githubusercontent.com/81459293/154806950-e7776d79-256e-4927-a00b-0b1dc706b2fe.png)

#### Tenure

![image](https://user-images.githubusercontent.com/81459293/154807308-1135d3e9-3db4-44e5-8403-cb1ae2791b78.png)

![image](https://user-images.githubusercontent.com/81459293/154807333-015d370e-540a-4c02-90c2-8d6d89c32934.png)

#### Monthly Charges

![image](https://user-images.githubusercontent.com/81459293/154807432-6c9e6a50-6431-4a79-8706-9b92fe409a9f.png)

![image](https://user-images.githubusercontent.com/81459293/154807453-bf7532fd-634b-4171-bfe5-5e05f119d52d.png)

As seen in the graphs for the different numerical features while tenure is slightly linear, the Mothly Charges and Total Charges are not and hence , simply using numerical data might affects the weights for that features as it contributes to the model differently at different ranges. Hence bucketing was done where new catergorical features were engineered based on different effects that different ranges have on the percetage of people who defaulted.

## Evaluation matrix

## Model Selection

# How to use

## API

## Docker

# Refrences
Facets:https://pair-code.github.io/facets/
TabNet:https://github.com/titu1994/tf-TabNet
