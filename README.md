# Accelerating Stochastic Gradient Descent with Anderson Acceleration

## Project Overview

This project explores the acceleration of Stochastic Gradient Descent (SGD) using Anderson Acceleration (AA) in two forms: one using normal equations and the other using QR updating. The focus is on applying these methods to simple linear regression, polynomial regression, and logistic regression problems.

The project demonstrates the implementation of these algorithms and compares their performance in terms of convergence rate, accuracy, and computational efficiency. The comparisons are based on synthetic datasets generated for each regression type.

## Results Summary

The results from running the experiments highlight the following:

- **Simple Linear Regression**: 
  - SGD with AA using normal equations converged the fastest, needing only two iterations to almost converge.
  - AA using QR was close behind, converging by the seventh iteration.
  - Unaccelerated SGD had the slowest convergence but the fastest execution time, likely due to the lower computational complexity per iteration.

- **Polynomial Regression**:
  - AA with QR outperformed other methods, converging in just two or three iterations.
  - Unaccelerated SGD was initially competitive but required about 30 iterations to fully converge.
  - AA using normal equations showed a slight delay in convergence at certain points but converged by the eighth iteration.

- **Logistic Regression**:
  - The convergence was slower across all methods, with AA using normal equations reaching convergence within the first 100 iterations.
  - Both unaccelerated SGD and AA with QR followed a similar pattern, converging near 2000 iterations.
  - Regularized least squares (RLS) method was faster in execution but less accurate, showing the highest norm-wise relative backward error.

Overall, the choice of the best SGD variant depends on the problem's requirements:
- **Fast convergence**: AA with normal equations is preferable.
- **High accuracy**: AA with QR is recommended.
- **Fast execution with acceptable accuracy**: The unaccelerated or regularized approach may be the best choice.

## Files Description

- **`datasets.py`**: Generates synthetic datasets for linear, polynomial, and logistic regression.
- **`sgd.py`**: Implements basic SGD for the regression problems.
- **`sgd_aa_norm.py`**: Implements SGD with Anderson Acceleration using normal equations.
- **`sgd_aa_qr.py`**: Implements SGD with Anderson Acceleration using QR updating.
- **`sgd_comp.py`**: Compares the performance of unaccelerated SGD with the two accelerated versions.
- **`sgd_comp_normalized.py`**: Compares the performance of regularized least squares with unaccelerated SGD and AA-accelerated versions.