# Assignment 8 – Artificial Neural Networks (ANN) for Classification

## Objective

The objective of this assignment is to build and evaluate an Artificial Neural Network (ANN) model for a classification problem. The model is trained using TensorFlow/Keras to learn patterns from the dataset and predict the target class accurately. The assignment also aims to understand the architecture of neural networks, the role of hidden layers, and the advantages and limitations of deep learning models.

---

## Dataset Link

Kaggle Dataset:
https://www.kaggle.com/

> Note: The dataset is **not included** in this repository in accordance with the assignment instructions. Please download it from the Kaggle link above.

---

## Libraries Used

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras

---

## Methodology

1. Load the dataset using Pandas.
2. Explore the dataset and check for missing values.
3. Separate input features and target variable.
4. Encode categorical variables (if required).
5. Split the dataset into training and testing sets.
6. Standardize the numerical features using StandardScaler.
7. Build an Artificial Neural Network (ANN) using TensorFlow/Keras.
8. Train the model using the training dataset.
9. Evaluate the model on the testing dataset.
10. Visualize the training and validation accuracy/loss.
11. Analyze the model performance and draw conclusions.

---

## Model Architecture

The ANN model consists of:

- Input Layer
  - Accepts the standardized input features.

- Hidden Layer(s)
  - Dense (Fully Connected) layer(s)
  - ReLU activation function
  - Learns complex and non-linear relationships from the data.

- Output Layer
  - Dense layer
  - Sigmoid activation (Binary Classification) or
  - Softmax activation (Multi-class Classification)

- Loss Function
  - Binary Crossentropy / Categorical Crossentropy

- Optimizer
  - Adam Optimizer

- Evaluation Metric
  - Accuracy

---

## Results

- Successfully trained the ANN model.
- Generated training and validation accuracy curves.
- Generated training and validation loss curves.
- Evaluated the model using the test dataset.
- The ANN achieved satisfactory classification performance and demonstrated its ability to learn complex relationships within the dataset.

---

## Conclusion

This assignment demonstrated how Artificial Neural Networks (ANNs) can be used to solve classification problems effectively. The trained model successfully learned patterns from the dataset and produced accurate predictions on unseen data. Hidden layers played a crucial role by extracting complex and non-linear features that improve the model's learning capability. Compared to traditional Machine Learning algorithms, Deep Learning can automatically learn hierarchical feature representations without requiring extensive manual feature engineering. However, ANNs require a large amount of training data and computational resources, and they are often more difficult to interpret than traditional models. Overall, ANN is a powerful approach for solving complex real-world classification problems.

---

## Repository Contents

```
Assignment-8.ipynb
README.md
```

---

## Note

As per the assignment instructions, the dataset has **not been uploaded** to this repository. Please download it directly from the Kaggle dataset link provided above.
