# Salary Prediction using Polynomial Regression

## Objective
Predict employee salary from position level using Polynomial Regression (degree=3).

## Dataset
https://www.kaggle.com/datasets/akram24/position-salaries

## Libraries Used
- pandas
- matplotlib
- scikit-learn

## Methodology
- Load dataset
- Check missing values
- Split data (80:20)
- Apply PolynomialFeatures (degree=3)
- Train Linear Regression on transformed features
- Evaluate using MAE, MSE and R²

## Results
- MAE: 70635.25
- MSE: 6263853282.86
- R² Score: 0.8763

## Conclusion
Polynomial Regression captures the nonlinear relationship between position level and salary better than simple Linear Regression. It produces accurate salary predictions for this dataset, though higher polynomial degrees may overfit.
