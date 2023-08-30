# Credit_Risk_Assessment_using_Random_Forest

**Title: Credit Risk Assessment using Random Forest: From Data Preprocessing to Model Evaluation**

**Introduction:**
In this code example, we perform credit risk assessment using a Random Forest classifier. The dataset contains information about individuals' credit profiles. We'll walk through the key steps involved, from importing libraries to evaluating the model's performance.

**Kaggle Dataset:**[Credit Risk Analysis](https://www.kaggle.com/datasets/nanditapore/credit-risk-analysis)

**Kaggle Notebook:**[Credit Risk Assessment Using Random Forest](https://www.kaggle.com/code/nanditapore/credit-risk-assessment-using-random-forest)

**Importing Libraries:**
The necessary libraries are imported, including Pandas for data handling, NumPy for numerical operations, Matplotlib for visualization, and scikit-learn's modules for model selection, ensemble methods, metrics, and imputation.

**Load and Preprocess the Dataset:**
The dataset is loaded from a CSV file. To prepare the data for modeling, the 'Id' column (unique identifier) is dropped, and categorical variables ('Home', 'Intent', 'Default') are one-hot encoded using Pandas' `get_dummies` function.

**Handle Missing Values with SimpleImputer:**
Missing values in the dataset are handled using `SimpleImputer`. The imputer fills missing values with the mean of each column, though other strategies like median or most frequent could also be used.

**Split Data into Features and Target:**
The data is split into features (X) and the target variable (y). The target variable is 'Status', indicating whether a credit application was approved or not. Additionally, the dataset is further split into training and testing sets using `train_test_split`.

**Initialize and Train the Random Forest Classifier:**
A Random Forest classifier is initialized with 100 estimators (decision trees) and a random seed of 42 for reproducibility. The model is then trained on the training data using the `fit` method.

**Make Predictions and Evaluate the Model:**
The trained model is used to predict the target values on the test set (`X_test`). The predictions (`y_pred`) are then evaluated.

**Visualize Random Forest Feature Importance:**
Feature importances from the trained Random Forest classifier are extracted. A horizontal bar plot is generated to visualize these feature importances, helping us understand which features have the most impact on the model's predictions.

**Visualize Confusion Matrix:**
A confusion matrix is created to visualize the model's performance on the test set. It shows the number of true positive, true negative, false positive, and false negative predictions. The Seaborn library is used to create a heatmap for a clearer representation.

**Print Accuracy and Classification Report:**
The model's accuracy is calculated using the `accuracy_score` function, comparing the predicted labels (`y_pred`) with the true labels (`y_test`). Additionally, a detailed classification report is generated using the `classification_report` function. This report includes precision, recall, and F1-score for each class, as well as macro and weighted averages across classes.

**Conclusion:**
The presented code showcases a comprehensive process for credit risk assessment using a Random Forest classifier. It highlights data preprocessing steps, model training, prediction, and evaluation through visualizations and metrics. The achieved accuracy and classification report provide insights into the model's performance on predicting credit risk.
