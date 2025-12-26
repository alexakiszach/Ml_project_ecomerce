# Data science Late delivery prediction:Project Overview
The primary goal of this project is to build and evaluate machine learning models to accurately predict whether an e-commerce order will reach the customer on time (Target_On_Time = 1 or 0).
##  Codes and resources Used
Python Version: 3.13.2
Packages:pandas,numpy,seaborn,matplotlib,sklearn

## Project steps
The project systematically compares several common classification algorithms to find the optimal model and hyperparameters.
EDA:Exploratory data analysis was completed.
Categorical Encoding: Nominal features (Warehouse_block, Mode_of_Shipment, Product_importance, Gender) were converted to numerical format using Label Encoding.
Feature Creation: New ratio features were engineered to potentially improve model performance.
Data Split  The data was split into training and testing sets. 
Feature Scaling: All features were scaled using StandardScaler to normalize the data distribution, which is crucial for distance-based algorithms (like KNN and SVM) and improves convergence for others.
## EDA
Exploratory data analysis was completed.Here are some key insights:
![newww](https://github.com/user-attachments/assets/c0c76c64-0f4f-4334-89bc-71fec2b2d07c)

![jpg](https://github.com/user-attachments/assets/4e868345-884a-4d29-b637-805ace3f77b1)
## Model Building
Created the following models:Decicion tree,K-Nearest neighbor,Support Vector Machine,Random Forest Classifier, XGBoost,Logistig Regression.The models then were evaluated, based on their accuracy,f1 score, precicion,recall and AUC.Accuracy was used as the main metric. The baseline metrics of the SVM algorithm, brought the best accuracy of 0.68
SVM:0.68
Random Forest Classifier: 0.65
Logistic Regression:0.66
KNN:0.66
Decicion Tree: 0.64
XGBoost:0.65

## Hyperparameter tuning
Hyperparameter tuning was done using GridSearchCV and RandomizedSearchCV. After that i reevaluated the metrics:
SVM:0.68
Random Forest Classifier: 0.68
Logistic Regression:0.66
KNN:0.66
Decicion Tree: 0.68
XGBoost:0.68
## Model Performance 
Based on Accuracies, every model seem to have the same accuracy score after hyperparameter tuning.
Although, depending on the business problem, some metrics will be key, like precicion, if we would want to classify the positive class with certainty.



## Dataset was used from kaggle : https://www.kaggle.com/datasets/prachi13/customer-analytics
