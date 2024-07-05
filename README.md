# Spam-Email-Classifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from google.colab import files

# Load the dataset
file_path = files.upload()

filename = list(file_path.keys())[0]

# Load the dataset from the uploaded CSV file
df = pd.read_csv(filename)

# Drop the "Email No." column as it is an identifier
df.drop(columns=['Email No.'], inplace=True)

# Sample 300 rows from the dataset
df = df.sample(n=300, random_state=42)

# Initialize the scaler
scaler = StandardScaler()

# Get the column names excluding 'Prediction'
columns_to_scale = df.columns.difference(['Prediction'])

# Scale the features
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Display the first few rows of the preprocessed dataframe
print("First few rows of the preprocessed dataframe:")
print(df.head())

# Split the dataset into features and target variable
X = df.drop(columns=['Prediction'])
y = df['Prediction']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
log_reg = LogisticRegression(max_iter=10000)
decision_tree = DecisionTreeClassifier()
svm = SVC()

# Function to display model performance
def display_performance(model_name, y_test, y_pred):
    print(f"{model_name} Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print()

# Train and evaluate Logistic Regression
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
display_performance("Logistic Regression", y_test, y_pred_log_reg)

# Train and evaluate Decision Tree
decision_tree.fit(X_train, y_train)
y_pred_decision_tree = decision_tree.predict(X_test)
display_performance("Decision Tree", y_test, y_pred_decision_tree)

# Train and evaluate SVM
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
display_performance("SVM", y_test, y_pred_svm)

# Cross-validation
log_reg_cv = cross_val_score(log_reg, X, y, cv=5)
decision_tree_cv = cross_val_score(decision_tree, X, y, cv=5)
svm_cv = cross_val_score(svm, X, y, cv=5)

print("Cross-Validation Mean Scores:")
print(f"Logistic Regression CV Mean Score: {log_reg_cv.mean():.4f}")
print(f"Decision Tree CV Mean Score: {decision_tree_cv.mean():.4f}")
print(f"SVM CV Mean Score: {svm_cv.mean():.4f}")
print()

# Initialize ensemble models
random_forest = RandomForestClassifier()
gradient_boosting = GradientBoostingClassifier()

# Train and evaluate Random Forest
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
display_performance("Random Forest", y_test, y_pred_rf)

# Train and evaluate Gradient Boosting
gradient_boosting.fit(X_train, y_train)
y_pred_gb = gradient_boosting.predict(X_test)
display_performance("Gradient Boosting", y_test, y_pred_gb)

# Hyperparameter tuning using Grid Search for the best model
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7]
}
grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best Hyperparameters for Gradient Boosting:")
print(grid_search.best_params_)
