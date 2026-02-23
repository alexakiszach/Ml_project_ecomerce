# Data science Late delivery prediction:Project Overview
The primary goal of this project is to build and evaluate machine learning models to accurately predict whether an e-commerce order will reach the customer on time (Target_On_Time = 1 or 0).
## 💻 Tech Stack & Environment

This project was developed using the following environment and libraries:

### **Language & Version**
* **Python:** `3.13.2`

### **Libraries & Frameworks**
| Library | Purpose |
| :--- | :--- |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computing and array processing |
| **Matplotlib** | Core data visualization |
| **Seaborn** | Statistical data visualization |
| **Scikit-learn (sklearn)** | Machine Learning, preprocessing, and evaluation |




## 🛠️ Project Workflow & Methodology

The project follows a structured machine learning pipeline, from raw data exploration to final model optimization.

### 1. Exploratory Data Analysis (EDA)
Comprehensive EDA was conducted to understand feature distributions, identify outliers, and uncover correlations between shipping variables and delivery outcomes.

### 2. Preprocessing & Feature Engineering
To prepare the data for the algorithms, the following steps were taken:

* **Categorical Encoding:** Nominal features (`Warehouse_block`, `Mode_of_Shipment`, `Product_importance`, and `Gender`) were transformed using **Label Encoding**.
* **Feature Creation:** Engineered new ratio-based features to capture complex relationships between variables, enhancing the predictive power of the models.
* **Feature Scaling:** Applied **StandardScaler** to normalize the data. This was a critical step to ensure distance-based models like **KNN** and **SVM** performed correctly and to speed up convergence for **Logistic Regression**.



### 3. Data Splitting
The processed dataset was partitioned into **Training** and **Testing** sets to ensure that model evaluation was performed on unseen data, preventing overfitting.

### 4. Model Selection & Tuning
As detailed in the results section, we systematically compared:
* Distance-based models (SVM, KNN)
* Linear models (Logistic Regression)
* Ensemble/Tree models (Random Forest, XGBoost, Decision Tree)

Hyperparameter optimization was then executed via **GridSearchCV** and **RandomizedSearchCV**..
## EDA
Exploratory data analysis was completed.Here are some key insights:
![newww](https://github.com/user-attachments/assets/c0c76c64-0f4f-4334-89bc-71fec2b2d07c)

![jpg](https://github.com/user-attachments/assets/4e868345-884a-4d29-b637-805ace3f77b1)
## Model Building
## 🤖 Model Comparison & Evaluation

I trained and evaluated six different Machine Learning models. While I tracked **F1 Score, Precision, Recall, and AUC**, **Accuracy** was used as the primary metric to determine the best performing model.

## 📈 Results Summary
The **Support Vector Machine (SVM)** achieved the highest baseline accuracy.

| Model | Accuracy |
| :--- | :---: |
| **Support Vector Machine (SVM)** | **0.68** |
| Logistic Regression | 0.66 |
| K-Nearest Neighbor (KNN) | 0.66 |
| Random Forest Classifier | 0.65 |
| XGBoost | 0.65 |
| Decision Tree | 0.64 |

### 🛠️ Models Created
* **Decision Tree**
* **K-Nearest Neighbor (KNN)**
* **Support Vector Machine (SVM)**
* **Random Forest Classifier**
* **XGBoost**
* **Logistic Regression**

> **Conclusion:** The SVM model is the top performer for this dataset with a baseline accuracy of **0.68**.

## ⚙️ Hyperparameter Tuning & Final Results

To optimize the models, I performed hyperparameter tuning using **GridSearchCV** and **RandomizedSearchCV**. This process refined the model parameters to achieve the highest possible predictive power.

### 📊 Final Model Performance
After tuning, multiple models reached a top accuracy of **0.68**. 

| Model | Baseline Accuracy | Tuned Accuracy | Status |
| :--- | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** | 0.68 | **0.68** | — |
| **Random Forest Classifier** | 0.65 | **0.68** | 📈 Improved |
| **XGBoost** | 0.65 | **0.68** | 📈 Improved |
| **Decision Tree** | 0.64 | **0.68** | 📈 Improved |
| Logistic Regression | 0.66 | 0.66 | — |
| K-Nearest Neighbor (KNN) | 0.66 | 0.66 | — |



### 🛠️ Optimization Techniques
* **GridSearchCV:** Used for an exhaustive search over specified parameter values for the models.
* **RandomizedSearchCV:** Used for a fixed number of parameter settings sampled from specified distributions (more efficient for complex models like XGBoost).

> **Final Analysis:** Hyperparameter tuning was highly effective for the Tree-based models (Random Forest, XGBoost, and Decision Tree), bringing their performance up to match the SVM baseline of **0.68**.
## Model Performance 
Based on Accuracies, every model seem to have the same accuracy score after hyperparameter tuning.
Although, depending on the business problem, some metrics will be key, like precicion, if we would want to classify the positive class with certainty.



## Dataset was used from kaggle : https://www.kaggle.com/datasets/prachi13/customer-analytics
