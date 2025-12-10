The primary goal of this project is to build and evaluate machine learning models to accurately predict whether an e-commerce order will reach the customer on time (Target_On_Time = 1 or 0).

The project addresses class imbalance using the ADASYN oversampling technique and systematically compares several common classification algorithms to find the optimal model and hyperparameters.
1. Data Preparation and Feature Engineering
The initial dataset was loaded and preprocessed through the following steps:

Categorical Encoding: Nominal features (Warehouse_block, Mode_of_Shipment, Product_importance, Gender) were converted to numerical format using Label Encoding.

Feature Creation: Three new ratio features were engineered to potentially improve model performance.

Data Split and Resampling: The data was split into training and testing sets. The training data was then balanced using the ADASYN (Adaptive Synthetic Sampling) technique to ensure models do not become biased toward the majority class (on-time delivery).


Feature Scaling: All features were scaled using StandardScaler to normalize the data distribution, which is crucial for distance-based algorithms (like KNN and SVM) and improves convergence for others.
DATASET USED FOR THE CLASSIFICATION PROBLEM : https://www.kaggle.com/datasets/prachi13/customer-analytics
