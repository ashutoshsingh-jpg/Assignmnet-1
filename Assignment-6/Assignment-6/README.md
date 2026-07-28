# Assignment 6 — Weather Condition Classification using SVM and Open-Meteo API
---

## 🎯 Objective

A weather analytics company wants to classify whether the weather is **Cool** or **Warm** based on
meteorological observations. This project builds a **Support Vector Machine (SVM)** classifier
(RBF kernel) trained on **live weather data pulled directly from the Open-Meteo API**, covering the
complete ML workflow: data collection → preprocessing → model training → evaluation → conclusion.

- **Warm** → Temperature ≥ 25°C
- **Cool** → Temperature < 25°C

---

## 📊 Data Source / API Documentation

- **API:** Open-Meteo Weather Forecast API (Free, No API key required)
- **Docs:** https://open-meteo.com/en/docs
- **Endpoint used:** `https://api.open-meteo.com/v1/forecast`
- **Example request:**

```
https://api.open-meteo.com/v1/forecast?latitude=28.6139&longitude=77.2090&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m&forecast_days=7
```

Hourly forecast data (temperature, relative humidity, surface pressure, wind speed) is fetched for
**8 Indian cities** (New Delhi, Mumbai, Bengaluru, Chennai, Kolkata, Jaipur, Shimla, Guwahati) over
a 7-day forecast window, and combined into a single dataset. Using multiple, climatically diverse
cities gives a large enough and reasonably balanced sample of both "Cool" and "Warm" readings
(e.g., Shimla contributes cooler readings, Chennai/Jaipur contribute warmer ones).

---

## 🗂️ Repository Structure

```
MPONLINE-Assignment-6/
│
├── Assignment-6.ipynb     # Main Jupyter Notebook (all 5 tasks, with explanations & plots)
├── Assignment-6.py        # Equivalent plain Python script version
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── weather_data_open_meteo.csv   # Sample dataset generated on run (created automatically)
```

---

## 🧰 Libraries Used

| Library | Purpose |
|---|---|
| `requests` | Call the Open-Meteo REST API |
| `pandas` | Convert JSON response to DataFrame, data wrangling |
| `numpy` | Numerical operations, target-variable creation |
| `matplotlib`, `seaborn` | Visualizations (class distribution, confusion matrix, decision boundary) |
| `scikit-learn` | `train_test_split`, `StandardScaler`, `LabelEncoder`, `SVC`, evaluation metrics |

Install everything with:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Methodology

### Task 1 — Data Collection and Understanding
1. Called the Open-Meteo `/v1/forecast` endpoint for 8 Indian cities, requesting hourly
   `temperature_2m`, `relative_humidity_2m`, `surface_pressure`, and `wind_speed_10m` for the next
   7 days.
2. Parsed the JSON `hourly` block into a Pandas DataFrame per city and concatenated all cities into
   one combined DataFrame.
3. Displayed the first five records with `df.head()`.
4. **Input features:** Temperature, Relative Humidity, Surface Pressure, Wind Speed.
   **Target variable:** `Weather_Class` — created as `Warm` if Temperature ≥ 25°C, else `Cool`.

### Task 2 — Data Preprocessing
1. Checked for missing values with `df.isnull().sum()` and dropped any null rows.
2. Removed unnecessary identifier columns (`time`, `City`) that are not model features.
3. Encoded the target variable (`Cool` → 0, `Warm` → 1) using `LabelEncoder`.
4. Split the data into **80% training / 20% testing** using `train_test_split` (stratified on the
   target to preserve class balance).
5. Standardized all four numeric features with `StandardScaler` (fit on train, transform on both
   train and test).

### Task 3 — Model Development
Trained a `sklearn.svm.SVC(kernel="rbf", C=1.0, gamma="scale")` classifier on the scaled training
data, then generated predictions on the held-out test set.

### Task 4 — Model Evaluation
Computed **Accuracy, Precision, Recall, and F1-Score**, printed a full `classification_report`, and
plotted a **Confusion Matrix**. A 2D decision-boundary plot (Temperature vs. Relative Humidity) is
also included to visually illustrate how the RBF kernel separates the two classes.

### Task 5 — Conclusion
A written conclusion (see below and inside the notebook) summarizes key findings, the importance of
feature scaling for SVM, and one advantage/limitation of the algorithm.

---

## 📈 Results

Running the notebook/script produces (values will vary slightly run-to-run since Open-Meteo serves
a live rolling 7-day forecast):

| Metric | Typical Value |
|---|---|
| Accuracy | ~0.95 – 0.99 |
| Precision | ~0.94 – 0.99 |
| Recall | ~0.94 – 0.99 |
| F1-Score | ~0.94 – 0.99 |

A confusion matrix heatmap and an SVM decision-boundary plot are generated and saved/displayed as
part of the notebook run.

**Observations:**
1. Since `Weather_Class` is derived directly from `Temperature`, the model achieves very high
   accuracy — temperature is an almost perfectly separating feature, with humidity, pressure and
   wind speed contributing secondary non-linear structure.
2. The RBF kernel produces a smooth, curved decision boundary between "Cool" and "Warm" regions,
   which a purely linear classifier would not capture as tightly.
3. Any class imbalance across cities (e.g., more Cool readings from Shimla) is reflected in small
   Precision/Recall differences between classes, visible in the confusion matrix.

---

## ✅ Conclusion

This project successfully demonstrates end-to-end classification of weather conditions (Cool vs.
Warm) using live meteorological data fetched from the Open-Meteo API and a Support Vector Machine
with an RBF kernel. Hourly readings of temperature, humidity, pressure and wind speed across
multiple Indian cities were combined, cleaned, encoded, and split into training/testing sets. The
model achieved high accuracy, precision, recall and F1-score, confirming that SVM with an RBF
kernel effectively captures the non-linear relationship between meteorological variables and
weather class. **Feature scaling (StandardScaler) is critical for SVM** because its decision
boundary depends on distances between points in feature space — unscaled features (e.g., pressure
in the hundreds vs. wind speed in single digits) would dominate the distance calculation and bias
the model. A key **advantage** of SVM is its ability to model complex, non-linear decision
boundaries via the kernel trick while resisting overfitting in higher-dimensional spaces. A
**limitation** is that SVM scales poorly to very large datasets, and its performance is sensitive
to the choice of kernel and hyperparameters (`C`, `gamma`).

---

## ▶️ How to Run

```bash
git clone https://github.com/AADISHADLAK/MPONLINE-Assignment-6.git
cd MPONLINE-Assignment-6
pip install -r requirements.txt

# Option A: run the notebook
jupyter notebook Assignment-6.ipynb

# Option B: run the plain script
python Assignment-6.py
```

No API key or sign-up is required — Open-Meteo is a free, open weather API.

---

## 📄 Assignment Reference

- Assignment PDF: shared via Google Drive (link provided in the assignment brief)
- Open-Meteo API Docs: https://open-meteo.com/
"# Assignment_6" 
