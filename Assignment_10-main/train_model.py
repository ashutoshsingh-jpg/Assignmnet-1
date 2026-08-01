import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def main():
    # Set the working directory to the script's directory to ensure local file paths work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("=================== TASK 1: Data Preprocessing ===================")
    
    # 1. Load the dataset using Pandas
    csv_path = 'heart.csv'
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 2. Display the first five records
    print("\nFirst 5 records:")
    print(df.head())
    
    # 3. Identify: Numerical features and Target variable
    # Identify numerical features (excluding target)
    target_col = 'target' if 'target' in df.columns else df.columns[-1]
    numerical_features = df.select_dtypes(include=[np.number]).columns.drop(target_col).tolist()
    
    print("\nNumerical features identified:")
    print(numerical_features)
    print(f"\nTarget variable identified: {target_col}")
    
    # 4. Check for missing values
    print("\nChecking for missing values:")
    missing_vals = df.isnull().sum()
    print(missing_vals[missing_vals > 0] if missing_vals.sum() > 0 else "No missing values found.")
    
    # 5. Split the dataset into 80% training and 20% testing
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n=================== TASK 2: Model Development ===================")
    
    # Train Random Forest Classifier
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predictions and evaluation
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Evaluation (Accuracy Score): {accuracy:.4f} ({accuracy * 100:.2f}%)")
    
    # Save the model, scaler, and features list
    print("\nSerializing artifacts...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(list(X.columns), 'feature_names.pkl')
    print("Saved 'model.pkl', 'scaler.pkl', and 'feature_names.pkl' successfully.")
    
    print("\nTasks 1 and 2 completed successfully!")

if __name__ == '__main__':
    main()
