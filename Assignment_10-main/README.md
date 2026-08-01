# AI-ML Assignment 10 — End-to-End Machine Learning Model Deployment using GitHub and Render

## Student Submission Details

| Field                             | Details                                            |
| --------------------------------- | -------------------------------------------------- |
| **Name**                          | ASHUTOSH SINGH                              |
| **Registration Number**           | 23BCE11453                                       |
| **Application Number**            | IN26011517                                        |
| **Batch Number**                  | 2B                                           |
| **Assignment Number**             | Assignment - 10                                    |
| **Email Address**                 | ashutoshsinghshiva57@gmail.com             |
| **Public GitHub Repository Link** | https://github.com/ashutoshsingh-jpg/Assignmnet-1/edit/main/Assignment_10-main/README.md   |
| **Render Deployment URL**         | `https://heartdiseasedeployment-kswe.onrender.com` |

---

## 🎯 Problem Statement

A healthcare organization wants to deploy a machine learning model that predicts whether a patient is at risk of heart disease based on clinical parameters.

This project covers the full lifecycle of an ML application:

1. **Data Understanding & Preprocessing**: Loading, inspecting, splitting, and scaling the clinical data.
2. **Model Development**: Training a Random Forest classifier and evaluating it.
3. **API Development**: Creating a REST API using Flask to serve predictions over HTTP.
4. **Interactive Web Interface**: Designing a responsive glassmorphic frontend.
5. **Production Deployment**: Setting up version control and deploying the live service to Render.

---

## 📊 Dataset

- **Name**: Heart Disease Prediction Dataset
- **Source**: [Kaggle - johnsmith88/heart-disease-dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Features**:
  - `age` - Patient's age (in years)
  - `sex` - Biological sex (1 = male; 0 = female)
  - `cp` - Chest pain type (0 = typical angina; 1 = atypical angina; 2 = non-anginal pain; 3 = asymptomatic)
  - `trestbps` - Resting blood pressure (in mm Hg on admission)
  - `chol` - Serum cholesterol (in mg/dl)
  - `fbs` - Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
  - `restecg` - Resting electrocardiographic results (0 = normal; 1 = ST-T wave abnormality; 2 = left ventricular hypertrophy)
  - `thalach` - Maximum heart rate achieved
  - `exang` - Exercise induced angina (1 = yes; 0 = no)
  - `oldpeak` - ST depression induced by exercise relative to rest
  - `slope` - The slope of the peak exercise ST segment (0 = upsloping; 1 = flat; 2 = downsloping)
  - `ca` - Number of major vessels (0-4) colored by fluoroscopy
  - `thal` - Thalassemia (1 = fixed defect; 2 = normal; 3 = reversible defect)
  - `target` - Diagnosis of heart disease (1 = normal; 2 = heart disease detected)

---

## 🛠️ Repository Structure

```text
Assignment_10/ (Repository Root)
├── app.py                  # Flask Application serving the API & UI
├── train_model.py          # Script to preprocess data and train the classifier
├── heart.csv               # Clinical dataset (20 sample records)
├── model.pkl               # Serialized Random Forest model binary
├── scaler.pkl              # Serialized StandardScaler binary
├── feature_names.pkl       # Serialized list of feature names for order consistency
├── requirements.txt        # Project package dependencies
├── .gitignore              # Files to ignore in Git
├── README.md               # Assignment documentation (this file)
└── templates/
    └── index.html          # Interactive Web UI serving the form
```

---

## 🧭 Methodology

### Task 1: Data Understanding and Preprocessing

- **Loading**: Loaded the dataset using `pandas.read_csv('heart.csv')`.
- **Inspection**: Displayed the first 5 records using `.head()`.
- **Identification**: Identified 13 numerical columns as input features, and `target` as the target variable. Checked for missing values (0 found).
- **Splitting**: Split the 20-row dataset into an 80% training set (16 samples) and a 20% test set (4 samples) using `train_test_split`.
- **Scaling**: Standardized numerical features with `StandardScaler` to ensure all columns contribute equally to model evaluation.

### Task 2: Model Development

- Trained a `RandomForestClassifier(n_estimators=100)` on the scaled training features.
- Evaluated performance on the test set using the `accuracy_score` metric, achieving an accuracy of 50.0% (expected due to the very small test sample size of 4).
- Serialized the trained model (`model.pkl`), feature scaler (`scaler.pkl`), and feature list (`feature_names.pkl`) using `joblib`.

### Task 3: API Development

- Developed a **Flask REST API** in `app.py`.
- Implemented a `POST /predict` endpoint that:
  - Accepts patient parameters as JSON or Form inputs.
  - Aligns features in the correct order, scales them using the loaded `scaler.pkl`, and runs predictions via `model.pkl`.
  - Maps numerical predictions back to labels: `2` ➔ `"Heart Disease Detected"`, `1` ➔ `"No Heart Disease Detected"`.
  - Returns the output response as a JSON object, e.g.:
    ```json
    {
      "prediction": "Heart Disease Detected",
      "raw_prediction": 2
    }
    ```
- Implemented a beautiful, glassmorphic **web interface** (`templates/index.html`) using clean HTML, custom Outfit typography, and an AJAX-based form submission so users can test clinical parameters interactively without reloading.

---

## 📈 Results

- **Model**: Random Forest Classifier
- **Test Accuracy**: 0.5000 (50.00%) on the 4-record test set
- **Input Dimensions**: 13 features
- **Output Classes**: Binary prediction (Heart Disease Detected / No Heart Disease Detected)

---

## ✅ Task 5: Conclusion

In this assignment, a heart disease prediction model was trained and deployed as a REST API and glassmorphic web interface. While model accuracy was limited to 50.0% due to the extremely small dataset of twenty patient records, the end-to-end pipeline was successfully established. Key challenges faced during deployment included configuring directory dependencies, aligning scikit-learn serialization versions, and managing web service binding for port allocation on Render. This highlighted the crucial role of MLOps (Machine Learning Operations). In real-world projects, MLOps ensures reproducibility, automates version control for both code and data, and streamlines model monitoring and updates. By establishing robust CI/CD and deployment pipelines, MLOps bridges the gap between development and production, allowing organizations to serve predictions reliably at scale and continuously monitor models for drift or degradation.

---

## 🚀 How to Run Locally

### 1. Prerequisites

Make sure you have Python 3 installed.

### 2. Set Up a Virtual Environment (Optional)

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

Navigate to the project folder and run:

```bash
pip install -r requirements.txt
```

### 4. (Optional) Re-train the Model

To fit the model and re-generate the pickle files:

```bash
python train_model.py
```

### 5. Run the Flask Web Application

Start the Flask dev server:

```bash
python app.py
```

Open your browser and visit: `http://localhost:5000` to interact with the web form.

### 6. Test the API via Command Line (cURL)

You can test the prediction endpoint directly with a POST request:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}'
```

Expected output:

```json
{ "prediction": "No Heart Disease Detected", "raw_prediction": 1 }
```

"# Assignment_10"
