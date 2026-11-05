# src/data/preprocess.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Loads raw data, formats types, and removes unnecessary columns."""
    df = pd.read_csv(file_path)
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", np.nan), errors='coerce')
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df

def build_pipeline(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """Constructs the Scikit-learn preprocessing pipeline."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def execute_preprocessing(input_path: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    """Orchestrates data splitting, pipeline fitting, and artifact saving to prevent leakage."""
    df = load_and_clean_data(input_path)
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Split BEFORE preprocessing to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in X.columns if col not in numeric_features]
    
    preprocessor = build_pipeline(numeric_features, categorical_features)
    
    # Fit strictly on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    feature_names = (
        numeric_features + 
        preprocessor.named_transformers_['cat']
        .named_steps['onehot']
        .get_feature_names_out(categorical_features).tolist()
    )
    
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    os.makedirs(output_dir, exist_ok=True)
    X_train_df.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test_df.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.joblib'))

if __name__ == "__main__":
    execute_preprocessing(
        input_path="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        output_dir="data/processed/"
    )