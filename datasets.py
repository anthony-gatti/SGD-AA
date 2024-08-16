import numpy as np

def generate_linear_regression_data():
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    X_b = np.c_[np.ones((100, 1)), X]
    return X_b, y

def generate_polynomial_regression_data():
    np.random.seed(0)
    X = 6 * np.random.rand(100, 1) - 3
    y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)
    X_poly = np.c_[np.ones((100, 1)), X, X**2]
    return X_poly, y

def generate_logistic_regression_data():
    np.random.seed(0)
    X = 2 * np.random.rand(100, 2) - 1
    y = (3 * X[:, 0] + 4 * X[:, 1] + np.random.randn(100) > 0).astype(np.int)
    y = y.reshape(-1, 1)
    X_b = np.c_[np.ones((100, 1)), X]
    return X_b, y