
# Assignment 4 - Breast Cancer Classification using KNN

## Objective
Build a K-Nearest Neighbors (KNN) classifier to predict whether a breast tumor is Malignant or Benign.

## Dataset
Breast Cancer Wisconsin Diagnostic Dataset

https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

## Libraries Used
- pandas
- scikit-learn
- numpy

## Methodology
1. Load dataset.
2. Remove unnecessary columns (`id`, `Unnamed: 32`).
3. Encode target variable.
4. Standardize features.
5. Split data into 80% training and 20% testing.
6. Train a KNN classifier with K=5.
7. Evaluate using Accuracy, Precision, Recall, F1-score and Confusion Matrix.

## Results
- Accuracy: 0.9561
- Precision: 0.9744
- Recall: 0.9048
- F1 Score: 0.9383

Confusion Matrix:
[[71  1]
 [ 4 38]]

## Conclusion
The KNN classifier produced high accuracy in classifying breast tumors into malignant and benign categories. Standardizing the features was essential because KNN relies on distance calculations, and unscaled features can dominate the distance metric, leading to poor predictions. The model achieved excellent precision and recall, showing that it can effectively identify malignant tumors while minimizing incorrect predictions. However, one limitation of KNN is that prediction becomes slower for large datasets because it must calculate distances to all training samples. Choosing an appropriate value of K is also important for obtaining the best performance.
