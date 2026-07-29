# Assignment 7 - Customer Segmentation using K-Means Clustering and PCA

## Objective

The objective of this assignment is to segment mall customers into different groups using the K-Means Clustering algorithm based on their annual income and spending behavior. Principal Component Analysis (PCA) is then applied to reduce the data to two dimensions for better visualization and interpretation of the clusters.

---

## Dataset

**Dataset:** Mall Customer Segmentation Dataset

**Source:** Kaggle

https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

> **Note:** The dataset is not included in this repository as per the assignment instructions. Please download it from the Kaggle link above.

---

## Libraries Used

- Pandas
- NumPy
- Matplotlib
- Scikit-learn
  - StandardScaler
  - KMeans
  - PCA

---

## Methodology

1. Loaded the dataset using Pandas.
2. Performed data exploration and summary statistics.
3. Checked for missing values.
4. Removed the `CustomerID` column.
5. Encoded categorical features (if required).
6. Standardized numerical features using `StandardScaler`.
7. Used the Elbow Method to determine the optimal number of clusters.
8. Trained the K-Means clustering model.
9. Assigned cluster labels to each customer.
10. Applied PCA to reduce the dataset to two principal components.
11. Visualized the customer clusters using scatter plots.

---

## Results

- The Elbow Method indicated **5** as the optimal number of clusters.
- Customers were grouped based on similar annual income and spending behavior.
- PCA reduced the dataset to two dimensions while preserving the cluster structure.
- The visualizations clearly showed distinct customer segments for business analysis.

---

## Conclusion

This assignment demonstrated the use of K-Means Clustering for customer segmentation and PCA for dimensionality reduction. The model successfully identified meaningful customer groups that can help businesses develop targeted marketing strategies. K-Means is simple and efficient but requires the number of clusters to be specified in advance. PCA effectively simplifies high-dimensional data, making cluster visualization easier while retaining most of the important information.

---

## Repository Structure

```
Assignment-7.ipynb
README.md
```
