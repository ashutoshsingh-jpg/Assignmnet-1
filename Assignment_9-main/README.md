# Assignment 9 — Image Classification using CNN (Cats vs Dogs)

| Field                   | Details                                       |
| ----------------------- | --------------------------------------------- |
| **Name**                | ASHUTOSH SINGH                              |
| **Registration Number** | 23BCE11453                                  |
| **Application Number**  | IN26011517                                  |
| **Email**               | ashutosh.23bce11453@vitbhopal.ac.in            |
| **Assignment**          | Assignment - 9                                |
| **Topic**               | Image Classification using CNN (Cats vs Dogs) |

## Objective

To develop a Convolutional Neural Network (CNN) that automates the
classification of pet images into **Cats** and **Dogs**, helping an
animal welfare organization sort images accurately and efficiently.

## Dataset Link

Cats vs Dogs Dataset (Kaggle):
https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset

The dataset is downloaded programmatically via `kagglehub` (not
uploaded to this repository):

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")

print("Path to dataset files:", path)
```

## Libraries Used

- `tensorflow` / `keras` — model building, training, data generators
- `kagglehub` — dataset download
- `numpy` — numerical operations
- `matplotlib` — visualizations (sample images, accuracy/loss curves)
- `scikit-learn` — precision, recall, F1-score, confusion matrix
- `Pillow (PIL)` — image loading/inspection

## Methodology

1. **Data Understanding** — Downloaded the dataset via `kagglehub`,
   inspected the folder structure, counted classes/images, checked
   image dimensions, and visualized five sample images with labels.
   The dataset contains **2 classes** (Cat, Dog) with **12,499 images
   each**, totaling **24,998 images**, at varying original resolutions
   (e.g. 500×375, 440×440).
2. **Data Preprocessing** — Resized all images to 128×128, normalized
   pixel values to the 0–1 range, and split the data into 80% training
   (20,000 images) / 20% testing (4,998 images) using Keras'
   `ImageDataGenerator`.
3. **Model Development** — Built and trained a CNN (architecture
   below) for 10 epochs using the Adam optimizer and binary
   crossentropy loss.
4. **Model Evaluation** — Evaluated test accuracy, precision, recall,
   and F1-score; plotted the confusion matrix and accuracy/loss curves
   across epochs.
5. **Conclusion** — Summarized key findings and CNN characteristics.

## CNN Architecture

| Layer          | Details                |
| -------------- | ---------------------- |
| Conv2D         | 32 filters, 3×3, ReLU  |
| MaxPooling2D   | 2×2                    |
| Conv2D         | 64 filters, 3×3, ReLU  |
| MaxPooling2D   | 2×2                    |
| Conv2D         | 128 filters, 3×3, ReLU |
| MaxPooling2D   | 2×2                    |
| Flatten        | —                      |
| Dense          | 128 neurons, ReLU      |
| Output (Dense) | 1 neuron, Sigmoid      |

**Compilation:**

- Optimizer: Adam
- Loss: Binary Crossentropy
- Metric: Accuracy
- Epochs: 10

## Results

| Metric        | Value  |
| ------------- | ------ |
| Test Accuracy | 0.8285 |
| Precision     | 0.8698 |
| Recall        | 0.7727 |
| F1-Score      | 0.8184 |

Training accuracy rose from **61.9%** (epoch 1) to **98.8%** (epoch 9),
while validation accuracy peaked around **84.0%** (epoch 6) and
plateaued between 83–84% for the remaining epochs.

Generated plots (saved in the repo after running `Assignment-9.py` /
`Assignment-9.ipynb`):

- `sample_images.png` — five sample images with class labels
- `confusion_matrix.png` — confusion matrix on the test set
- `accuracy_vs_epoch.png` — training vs validation accuracy over epochs
- `loss_vs_epoch.png` — training vs validation loss over epochs

**Observations:**

1. Training accuracy climbed steadily and reached **98.8%** by epoch 9,
   confirming the CNN successfully learns discriminative features to
   tell cats and dogs apart.
2. Validation accuracy plateaued around **83–84%** from epoch 6 onward
   while training accuracy kept climbing, and validation loss rose from
   ~0.37 (epoch 3) to ~0.70 (epoch 9) — a clear sign of **overfitting**
   in the later epochs.
3. Precision (0.87) is noticeably higher than recall (0.77), meaning
   the model is more conservative about predicting the positive class
   and misses a meaningful share of true positives (higher false
   negative rate).
4. The gap between train and validation performance suggests dropout,
   data augmentation, or early stopping would likely improve
   generalization on unseen images.

## Conclusion

This assignment demonstrates the use of a Convolutional Neural Network
(CNN) to classify images of cats and dogs. The model, built with three
convolution-pooling blocks followed by dense layers, learned meaningful
spatial features directly from raw pixel data and achieved a test
accuracy of **82.85%** (precision 0.87, recall 0.77, F1-score 0.82)
after 10 epochs of training, despite showing signs of overfitting in
later epochs as training accuracy pulled ahead of validation accuracy.
Convolutional layers extract
local patterns such as edges, textures, and shapes, while pooling
layers reduce spatial dimensions, control overfitting, and make the
learned features more robust to small translations in the image.
Compared to a traditional Artificial Neural Network (ANN), a CNN offers
the key advantage of parameter sharing and local connectivity, allowing
it to process images far more efficiently than a fully connected
network of comparable depth. One limitation of CNNs, however, is that
they typically require large amounts of labeled training data and
computational resources, and can still struggle with images that
differ significantly from the training distribution.

## How to Run

```bash
pip install tensorflow kagglehub scikit-learn matplotlib pillow
python Assignment-9.py
```

## Note

The dataset is **not** included in this repository. It is downloaded
at runtime via `kagglehub`, per the Kaggle dataset's license terms —
see the dataset link above.
"# Assignment_9"
