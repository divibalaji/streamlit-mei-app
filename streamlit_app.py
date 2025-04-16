import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy import stats
from statsmodels.api import qqplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

import streamlit as st
uploaded_file = st.file_uploader('Upload your CSV file', type='csv')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
st.write(df.head())

st.write(df.shape)

st.write(df.info())

st.write(df.describe().T)

st.write(df.columns)

st.write(df['sex'].value_counts())

st.write(df['cp'].value_counts())

st.write(df['target'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# Step 0: Column definitions
numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Step 1: Handle missing values
X = df.drop(['target'], axis=1)
y = df['target']

# Impute missing values
num_imputer = SimpleImputer(strategy='mean')
X[numerical_columns] = num_imputer.fit_transform(X[numerical_columns])

cat_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_columns] = cat_imputer.fit_transform(X[categorical_columns])

# Step 2: Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Step 4: Preprocessing with scaling + encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

st.write("Before PCA - X_train shape:", X_train_processed.shape)

# Step 5: PCA after full preprocessing
pca = PCA(n_components=13)
X_train_pca = pca.fit_transform(X_train_processed)
X_test_pca = pca.transform(X_test_processed)

# Step 6: Print info
st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
st.write("Total PCA Components selected:", pca.n_components_)
st.write("After PCA - X_train_pca shape:", X_train_pca.shape)


preprocessor.named_transformers_['cat'].get_feature_names_out()

st.write(preprocessor.get_feature_names_out())

import seaborn as sns
import matplotlib.pyplot as plt

# Box plot for a single numerical column
sns.boxplot(x=df['age'])  # Replace 'age' with your desired column
plt.title('Box Plot of Age')
plt.show()

# Box plots for multiple numerical columns
sns.boxplot(data=df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])  # Replace with your desired columns
plt.title('Box Plots of Numerical Features')
plt.show()

import pandas as pd

def remove_outliers_iqr(df, columns):
    """Removes outliers from specified columns using the IQR method.

    Args:
        df: pandas DataFrame
        columns: List of columns to remove outliers from

    Returns:
        DataFrame with outliers removed
    """

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Specify the columns containing numerical features with potential outliers
numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Remove outliers from the DataFrame
df_no_outliers = remove_outliers_iqr(df, numerical_columns)

# Print the shape of the original and cleaned DataFrames
st.write("Original DataFrame shape:", df.shape)
st.write("DataFrame shape after outlier removal:", df_no_outliers.shape)

import seaborn as sns
import matplotlib.pyplot as plt

# Box plots before outlier removal
sns.boxplot(data=df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])
plt.title('Box Plots Before Outlier Removal')
plt.show()

# Box plots after outlier removal
sns.boxplot(data=df_no_outliers[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])
plt.title('Box Plots After IQR Outlier Removal')
plt.show()

import pandas as pd
import numpy as np

def remove_outliers_zscore(df, columns, threshold=3):
    """Removes outliers from specified columns using the Z-score method.

    Args:
        df: pandas DataFrame
        columns: List of columns to remove outliers from
        threshold: Z-score threshold for outlier detection (default=3)

    Returns:
        DataFrame with outliers removed
    """

    for col in columns:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        df = df[(z_scores < threshold)]
    return df

# Specify the columns containing numerical features with potential outliers
numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Remove outliers from the DataFrame using Z-score method
df_no_outliers_zscore = remove_outliers_zscore(df, numerical_columns)

# Print the shape of the original and cleaned DataFrames
st.write("Original DataFrame shape:", df.shape)
st.write("DataFrame shape after outlier removal (Z-score):", df_no_outliers_zscore.shape)

import seaborn as sns
import matplotlib.pyplot as plt

# Box plots before outlier removal
sns.boxplot(data=df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])
plt.title('Box Plots Before Outlier Removal')
plt.show()

# Box plots after outlier removal (Z-score method)
sns.boxplot(data=df_no_outliers_zscore[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])
plt.title('Box Plots After Outlier Removal (Z-score)')
plt.show()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def remove_outliers_mad(df, columns, threshold=3):
    """Removes outliers from specified columns using the MAD method.

    Args:
        df: pandas DataFrame
        columns: List of columns to remove outliers from
        threshold: MAD threshold for outlier detection (default=3)

    Returns:
        DataFrame with outliers removed
    """

    for col in columns:
        median = df[col].median()
        mad = np.median(np.abs(df[col] - median))
        lower_bound = median - threshold * mad
        upper_bound = median + threshold * mad
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Specify the columns containing numerical features with potential outliers
numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Remove outliers from the DataFrame using MAD method
df_no_outliers_mad = remove_outliers_mad(df.copy(), numerical_columns)

# Print the shape of the original and cleaned DataFrames
st.write("Original DataFrame shape:", df.shape)
st.write("DataFrame shape after outlier removal (MAD):", df_no_outliers_mad.shape)

# Box plots before outlier removal
sns.boxplot(data=df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])
plt.title('Box Plots Before Outlier Removal')
plt.show()

# Box plots after outlier removal (MAD method)
sns.boxplot(data=df_no_outliers_mad[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])
plt.title('Box Plots After Outlier Removal (MAD)')
plt.show()

# 1. KDE Plot for Numerical Columns
def plot_kde(col_name):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x=col_name, hue="target", fill=True, palette="bright")
    plt.title(f"KDE Plot of {col_name} by Target")
    plt.xlabel(col_name)
    plt.ylabel("Density")
    plt.show()

# 2. Count Plot for Categorical Columns
def plot_count(col_name):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=col_name, hue="target", palette="viridis")
    plt.title(f"Count Plot of {col_name} by Target")
    plt.xlabel(col_name)
    plt.ylabel("Count")
    plt.show()

# Plotting for all numerical columns
for col in numerical_columns:
    plot_kde(col)

# Plotting for all categorical columns
for col in categorical_columns:
    plot_count(col)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 1: Feature Scaling(standardize data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical_columns])
X_test_scaled = scaler.transform(X_test[numerical_columns])

# Step 2: Covariance Matrix, Eigenvalues, and Eigenvectors
# Calculate the covariance matrix of the scaled features
cov_matrix = np.cov(X_train_scaled, rowvar=False)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Display the covariance matrix, eigenvalues, and eigenvectors
st.write("Covariance Matrix:\n", cov_matrix)
st.write("Eigenvalues:\n", eigenvalues)
st.write("Eigenvectors:\n", eigenvectors)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 1: Feature Scaling(standardize data)
# Scale all features (numerical + encoded categorical)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Optional: show shape before PCA
st.write("Before PCA - X_train shape:", X_train_scaled.shape)

# Step 2: Apply PCA directly after scaling
pca = PCA(n_components=13)  # retain 95% variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Output PCA info
st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
st.write("Total PCA Components selected:", pca.n_components_)
st.write("After PCA - X_train_pca shape:", X_train_pca.shape)

st.write(X_train_pca.shape)

import matplotlib.pyplot as plt
import numpy as np

# Fit PCA without reducing components first
pca_full = PCA()
pca_full.fit(X_train_scaled)

# Plot explained variance ratio
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o', linestyle='--', color='b')
plt.title('Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance Threshold')
plt.axvline(x=np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.95)+1, color='g', linestyle='--', label='Optimal Components')
plt.legend()
plt.tight_layout()
plt.show()


# Combining PCA Features with Categorical Features
# Convert PCA-transformed features into DataFrames

pca_columns = [f"PC{i+1}" for i in range(pca.n_components_)]
X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_columns, index=X_train.index)
X_test_pca_df = pd.DataFrame(X_test_pca, columns=pca_columns, index=X_test.index)

# Get categorical features
X_train_cat = X_train[categorical_columns]
X_test_cat = X_test[categorical_columns]

# Combine PCA features with categorical features
X_train_combined = pd.concat([X_train_pca_df.reset_index(drop=True), X_train_cat.reset_index(drop=True)], axis=1)
X_test_combined = pd.concat([X_test_pca_df.reset_index(drop=True), X_test_cat.reset_index(drop=True)], axis=1)

# Display shapes to verify
st.write("Original Training Data Shape (Scaled):", X_train_scaled.shape)
st.write("Original Test Data Shape (Scaled):", X_test_scaled.shape)
st.write("PCA-Transformed Training Data Shape:", X_train_pca.shape)
st.write("PCA-Transformed Test Data Shape:", X_test_pca.shape)

# Parameter grid for Logistic Regression
param_grid_lr = {
    'C': [0.01],           # Regularization strength
    'solver': ['liblinear', 'lbfgs']         # Solvers to handle different types of datasets
}

# Step 2: Hyperparameter tuning for Logistic Regression on original features
grid_search_lr = GridSearchCV(LogisticRegression(random_state=42), param_grid_lr, cv=5, scoring='accuracy')
grid_search_lr.fit(X_train, y_train)
best_lr_original = grid_search_lr.best_estimator_

# Logistic Regression on original features
y_pred_lr_original = best_lr_original.predict(X_test)
st.write("Logistic Regression on Original Features by Hyperparameter Tuning:")
st.write("Best Parameters:", grid_search_lr.best_params_)
st.write("Accuracy:", accuracy_score(y_test, y_pred_lr_original))
st.write("Precision:", precision_score(y_test, y_pred_lr_original))
st.write("Recall:", recall_score(y_test, y_pred_lr_original))
st.write("F1 Score:", f1_score(y_test, y_pred_lr_original))

# Confusion Matrix for Logistic Regression on Original Features
cm_lr_original = confusion_matrix(y_test, y_pred_lr_original)
ConfusionMatrixDisplay(cm_lr_original).plot()
plt.title("Confusion Matrix - Logistic Regression on Original Features")
plt.show()

# Classification Report for Logistic Regression on Original Features
st.write("\nClassification Report:")
st.write(classification_report(y_test, y_pred_lr_original))


# Hyperparameter tuning for Logistic Regression on PCA-transformed features
grid_search_lr_pca = GridSearchCV(LogisticRegression(random_state=42), param_grid_lr, cv=5, scoring='accuracy')
grid_search_lr_pca.fit(X_train_combined, y_train)
best_lr_pca = grid_search_lr_pca.best_estimator_

# Logistic Regression on PCA-transformed features
y_pred_lr_pca = best_lr_pca.predict(X_test_combined)
st.write("\nLogistic Regression on PCA-Transformed Features by Hyperparameter Tuning:")
st.write("Best Parameters:", grid_search_lr_pca.best_params_)
st.write("Accuracy:", accuracy_score(y_test, y_pred_lr_pca))
st.write("Precision:", precision_score(y_test, y_pred_lr_pca))
st.write("Recall:", recall_score(y_test, y_pred_lr_pca))
st.write("F1 Score:", f1_score(y_test, y_pred_lr_pca))

# Confusion Matrix for Logistic Regression on PCA-Transformed Features
cm_lr_pca = confusion_matrix(y_test, y_pred_lr_pca)
ConfusionMatrixDisplay(cm_lr_pca).plot()
plt.title("Confusion Matrix - Logistic Regression on PCA-Transformed Features")
plt.show()

# Classification Report for Logistic Regression on PCA-Transformed Features
st.write("\nClassification Report:")
st.write(classification_report(y_test, y_pred_lr_pca))


# Step 1: Define parameter grids for each model
from sklearn.ensemble import RandomForestClassifier

# Parameter grid for AdaBoost
param_grid_ada = {
    'n_estimators': [50, 100, 200, 300],          # Number of boosting rounds
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1]          # Learning rate
}

# Step 2: Hyperparameter tuning for AdaBoost on original features
grid_search_ada = GridSearchCV(AdaBoostClassifier(random_state=42), param_grid_ada, cv=5, scoring='accuracy')
grid_search_ada.fit(X_train, y_train)
best_ada_original = grid_search_ada.best_estimator_

# AdaBoost on original features
y_pred_ada_original = best_ada_original.predict(X_test)
st.write("\nAdaBoost on Original Features:")
st.write("Best Parameters:", grid_search_ada.best_params_)
st.write("Accuracy:", accuracy_score(y_test, y_pred_ada_original))
st.write("Precision:", precision_score(y_test, y_pred_ada_original))
st.write("Recall:", recall_score(y_test, y_pred_ada_original))
st.write("F1 Score:", f1_score(y_test, y_pred_ada_original))

# Confusion Matrix for AdaBoost on Original Features
cm_ada_original = confusion_matrix(y_test, y_pred_ada_original)
ConfusionMatrixDisplay(cm_ada_original).plot()
plt.title("Confusion Matrix - AdaBoost on Original Features")
plt.show()

# Classification Report for AdaBoost on Original Features
st.write("\nClassification Report:")
st.write(classification_report(y_test, y_pred_ada_original))


# Hyperparameter tuning for AdaBoost on PCA-transformed features
grid_search_ada_pca = GridSearchCV(AdaBoostClassifier(random_state=42), param_grid_ada, cv=5, scoring='accuracy')
grid_search_ada_pca.fit(X_train_combined, y_train)
best_ada_pca = grid_search_ada_pca.best_estimator_

# AdaBoost on PCA-transformed features
y_pred_ada_pca = best_ada_pca.predict(X_test_combined)
st.write("\nAdaBoost on PCA-Transformed Features:")
st.write("Best Parameters:", grid_search_ada_pca.best_params_)
st.write("Accuracy:", accuracy_score(y_test, y_pred_ada_pca))
st.write("Precision:", precision_score(y_test, y_pred_ada_pca))
st.write("Recall:", recall_score(y_test, y_pred_ada_pca))
st.write("F1 Score:", f1_score(y_test, y_pred_ada_pca))

# Confusion Matrix for AdaBoost on PCA-Transformed Features
cm_ada_pca = confusion_matrix(y_test, y_pred_ada_pca)
ConfusionMatrixDisplay(cm_ada_pca).plot()
plt.title("Confusion Matrix - AdaBoost on PCA-Transformed Features")
plt.show()

# Classification Report for AdaBoost on PCA-Transformed Features
st.write("\nClassification Report:")
st.write(classification_report(y_test, y_pred_ada_pca))


from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 1. Define parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 20],
    'min_samples_split': [2],
    'min_samples_leaf': [1, 2, 4]
}

# 2. Hyperparameter tuning for Random Forest on original features
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)
best_rf_original = grid_search_rf.best_estimator_

# Random Forest on original features
y_pred_rf_original = best_rf_original.predict(X_test)
st.write("\nRandom Forest on Original Features:")
st.write("Best Parameters:", grid_search_rf.best_params_)
st.write("Accuracy:", accuracy_score(y_test, y_pred_rf_original))
st.write("Precision:", precision_score(y_test, y_pred_rf_original))
st.write("Recall:", recall_score(y_test, y_pred_rf_original))
st.write("F1 Score:", f1_score(y_test, y_pred_rf_original))

# Confusion Matrix for Random Forest on Original Features
cm_rf_original = confusion_matrix(y_test, y_pred_rf_original)
ConfusionMatrixDisplay(cm_rf_original).plot()
plt.title("Confusion Matrix - Random Forest on Original Features")
plt.show()

# Classification Report for Random Forest on Original Features
st.write("\nClassification Report:")
st.write(classification_report(y_test, y_pred_rf_original))


# 3. Hyperparameter tuning for Random Forest on PCA-transformed features
grid_search_rf_pca = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf_pca.fit(X_train_combined, y_train)
best_rf_pca = grid_search_rf_pca.best_estimator_

# Random Forest on PCA-transformed features
y_pred_rf_pca = best_rf_pca.predict(X_test_combined)
st.write("\nRandom Forest on PCA-Transformed Features:")
st.write("Best Parameters:", grid_search_rf_pca.best_params_)
st.write("Accuracy:", accuracy_score(y_test, y_pred_rf_pca))
st.write("Precision:", precision_score(y_test, y_pred_rf_pca))
st.write("Recall:", recall_score(y_test, y_pred_rf_pca))
st.write("F1 Score:", f1_score(y_test, y_pred_rf_pca))

# Confusion Matrix for Random Forest on PCA-Transformed Features
cm_rf_pca = confusion_matrix(y_test, y_pred_rf_pca)
ConfusionMatrixDisplay(cm_rf_pca).plot()
plt.title("Confusion Matrix - Random Forest on PCA-Transformed Features")
plt.show()

# Classification Report for Random Forest on PCA-Transformed Features
st.write("\nClassification Report:")
st.write(classification_report(y_test, y_pred_rf_pca))



from sklearn.tree import DecisionTreeClassifier

# 1. Define parameter grid for Decision Tree
param_grid_dt = {
    'criterion': ['gini', 'entropy'],  # Splitting criterion
    'max_depth': [None, 5, 10, 15],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4]  # Minimum samples required at a leaf node
}

# 2. Hyperparameter tuning for Decision Tree on original features
grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, scoring='accuracy')
grid_search_dt.fit(X_train, y_train)
best_dt_original = grid_search_dt.best_estimator_

# Decision Tree on original features
y_pred_dt_original = best_dt_original.predict(X_test)
st.write("\nDecision Tree on Original Features:")
st.write("Best Parameters:", grid_search_dt.best_params_)
st.write("Accuracy:", accuracy_score(y_test, y_pred_dt_original))
st.write("Precision:", precision_score(y_test, y_pred_dt_original))
st.write("Recall:", recall_score(y_test, y_pred_dt_original))
st.write("F1 Score:", f1_score(y_test, y_pred_dt_original))

# Confusion Matrix for Decision Tree on Original Features
cm_dt_original = confusion_matrix(y_test, y_pred_dt_original)
ConfusionMatrixDisplay(cm_dt_original).plot()
plt.title("Confusion Matrix - Decision Tree on Original Features")
plt.show()

# 3. Hyperparameter tuning for Decision Tree on PCA-transformed features
grid_search_dt_pca = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, scoring='accuracy')
grid_search_dt_pca.fit(X_train_combined, y_train)
best_dt_pca = grid_search_dt_pca.best_estimator_

# Decision Tree on PCA-transformed features
y_pred_dt_pca = best_dt_pca.predict(X_test_combined)
st.write("\nDecision Tree on PCA-Transformed Features:")
st.write("Best Parameters:", grid_search_dt_pca.best_params_)
st.write("Accuracy:", accuracy_score(y_test, y_pred_dt_pca))
st.write("Precision:", precision_score(y_test, y_pred_dt_pca))
st.write("Recall:", recall_score(y_test, y_pred_dt_pca))
st.write("F1 Score:", f1_score(y_test, y_pred_dt_pca))

# Confusion Matrix for Decision Tree on PCA-Transformed Features
cm_dt_pca = confusion_matrix(y_test, y_pred_dt_pca)
ConfusionMatrixDisplay(cm_dt_pca).plot()
plt.title("Confusion Matrix - Decision Tree on PCA-Transformed Features")
plt.show()

!pip install streamlit pyngrok joblib


import joblib

# Assuming 'best_rf_original' is your desired model
# Replace 'best_rf_original' with the model you want to save (best_rf_pca, best_ada_original, etc.)
model_to_save = best_rf_original

joblib.dump(model_to_save, 'heart_disease_model.pkl')

%%writefile app.py
import streamlit as st
import numpy as np
import joblib

model = joblib.load('heart_disease_model.pkl')

st.title(" Cardiovascular Disease Prediction App")
st.write("Fill the form to check your heart health status:")

age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", options=[0, 1])
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol (mg/dl)", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
restecg = st.selectbox("Resting ECG Result", options=[0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1])
oldpeak = st.number_input("Oldpeak", value=1.0)
slope = st.selectbox("Slope of ST Segment", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-4)", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (1=Normal, 2=Fixed, 3=Reversible)", options=[1, 2, 3])

if st.button("Predict"):
    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(user_input)[0]

    if prediction == 1:
        st.error(" Risk of cardiovascular disease detected.")
    else:
        st.success(" No risk detected. Keep taking care!")


from pyngrok import ngrok

# Authenticate ngrok with your auth token

# Open a tunnel on port 8501 for Streamlit
public_url = ngrok.connect(8501)

# Output the public URL
st.write(f"Public URL: {public_url}")


!streamlit run app.py &>/dev/null &


!pip install shapash pyngrok


from pyngrok import ngrok

!ngrok config add-authtoken 2vlnWXWMdgCCYk89NEtCE4INpfm_3QTyPVwMZ2mc2mwAGctK3





