import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (7, 5)

BASE_URL = "https://api.open-meteo.com/v1/forecast"

# Cities used to build a diverse, sufficiently large dataset
CITIES = {
    "New Delhi":   (28.6139, 77.2090),
    "Mumbai":      (19.0760, 72.8777),
    "Bengaluru":   (12.9716, 77.5946),
    "Chennai":     (13.0827, 80.2707),
    "Kolkata":     (22.5726, 88.3639),
    "Jaipur":      (26.9124, 75.7873),
    "Shimla":      (31.1048, 77.1734),
    "Guwahati":    (26.1445, 91.7362),
}


def fetch_city_weather(city, lat, lon, forecast_days=7):
    """Fetch hourly weather data for a single city from the Open-Meteo API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "forecast_days": forecast_days,
    }
    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    hourly = data["hourly"]
    df_city = pd.DataFrame({
        "time": hourly["time"],
        "Temperature": hourly["temperature_2m"],
        "Relative_Humidity": hourly["relative_humidity_2m"],
        "Surface_Pressure": hourly["surface_pressure"],
        "Wind_Speed": hourly["wind_speed_10m"],
    })
    df_city["City"] = city
    return df_city


def main():
    # ---------------- Task 1: Data Collection and Understanding ----------------
    print("=" * 70)
    print("TASK 1: DATA COLLECTION AND UNDERSTANDING")
    print("=" * 70)

    all_frames = []
    for city, (lat, lon) in CITIES.items():
        try:
            all_frames.append(fetch_city_weather(city, lat, lon))
            print(f"Fetched {city}: OK")
        except Exception as e:
            print(f"Fetched {city}: FAILED ({e})")

    df = pd.concat(all_frames, ignore_index=True)
    print("\nDataset shape:", df.shape)
    print("\nFirst five records:\n", df.head())

    # Target variable
    df["Weather_Class"] = np.where(df["Temperature"] >= 25, "Warm", "Cool")
    print("\nClass distribution:\n", df["Weather_Class"].value_counts())

    # ---------------- Task 2: Data Preprocessing ----------------
    print("\n" + "=" * 70)
    print("TASK 2: DATA PREPROCESSING")
    print("=" * 70)

    print("\nMissing values per column:\n", df.isnull().sum())
    df = df.dropna().reset_index(drop=True)

    df_model = df.drop(columns=["time", "City"])

    label_encoder = LabelEncoder()
    df_model["Weather_Class_Encoded"] = label_encoder.fit_transform(df_model["Weather_Class"])
    print("\nEncoded classes:", list(label_encoder.classes_))

    feature_cols = ["Temperature", "Relative_Humidity", "Surface_Pressure", "Wind_Speed"]
    X = df_model[feature_cols]
    y = df_model["Weather_Class_Encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTraining set: {X_train.shape}, Testing set: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------- Task 3: Model Development ----------------
    print("\n" + "=" * 70)
    print("TASK 3: MODEL DEVELOPMENT (SVM - RBF Kernel)")
    print("=" * 70)

    svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    y_pred = svm_model.predict(X_test_scaled)
    print("Model trained successfully.")

    # ---------------- Task 4: Model Evaluation ----------------
    print("\n" + "=" * 70)
    print("TASK 4: MODEL EVALUATION")
    print("=" * 70)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - SVM (RBF Kernel)")
    plt.savefig("confusion_matrix.png", bbox_inches="tight")
    print("\nSaved confusion matrix plot to confusion_matrix.png")

    # Save the fetched dataset for reproducibility
    df.to_csv("weather_data_open_meteo.csv", index=False)
    print("Saved raw dataset to weather_data_open_meteo.csv")

    print("\n" + "=" * 70)
    print("TASK 5: CONCLUSION -> see README.md for the written conclusion")
    print("=" * 70)


if __name__ == "__main__":
    main()
