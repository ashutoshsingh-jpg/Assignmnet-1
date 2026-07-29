# Assignment 8 – Handwritten Digit Recognition using ANN

## Objective
Build an Artificial Neural Network (ANN) to classify handwritten digits (0–9) using the MNIST dataset.

## Dataset Link
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

## Libraries Used
- pandas
- numpy
- matplotlib
- scikit-learn
- tensorflow / keras

## Methodology
1. Load dataset
2. Normalize pixel values
3. One-hot encode labels
4. Train ANN (128-ReLU -> 64-ReLU -> 10-Softmax)
5. Evaluate using accuracy, confusion matrix and classification report.

## Model Architecture
Input(784) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)

## Results
The model is expected to achieve high accuracy (>97%) on the MNIST test set after 10 epochs.

## Conclusion
ANNs effectively recognize handwritten digits by learning hierarchical features. Hidden layers improve learning capacity, while the main limitation is higher computational cost.
