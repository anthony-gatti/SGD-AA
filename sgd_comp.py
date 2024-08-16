import numpy as np
import time
import matplotlib.pyplot as plt

# Generate synthetic data for linear regression
np.random.seed(0)
X = 2 * np.random.rand(100, 2) - 1
y = (3 * X[:, 0] + 4 * X[:, 1] + np.random.randn(100) > 0).astype(int)
y = y.reshape(-1, 1)
X_b = np.c_[np.ones((100, 1)), X]

# Add bias term
#X_b = np.c_[np.ones((100, 1)), X]

# Hyperparameters
learning_rate = 0.01
n_iterations = 2000
batch_size = 20
m_history = 5  # Anderson Acceleration history length

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Compute gradient function
def compute_gradient(X, y, theta):
    m = len(y)
    predictions = sigmoid(X.dot(theta))
    gradients = 1/m * X.T.dot(predictions - y)
    return gradients

# Anderson Acceleration update using normal equations
def anderson_acceleration_norm(theta_history, residuals_history):
    m = len(theta_history)
    if m == 0:
        return theta_history[-1]

    # Formulate the matrices G and F
    G = np.hstack(residuals_history)
    F = np.hstack(theta_history) - theta_history[-1]

    # Solve for the coefficients c
    try:
        c = np.linalg.lstsq(G.T @ G, G.T @ residuals_history[-1], rcond=None)[0]
    except np.linalg.LinAlgError:
        c = np.zeros((m, 1))

    # Compute the new update
    update = theta_history[-1] - F @ c

    return update

# Anderson Acceleration update using QR
def anderson_acceleration_qr(theta_history, residuals_history):
    m = len(theta_history)
    if m == 0:
        return theta_history[-1]

    # Formulate the matrices G and F
    G = np.hstack(residuals_history)
    F = np.hstack(theta_history) - theta_history[-1]

    # QR decomposition of G
    Q, R = np.linalg.qr(G)

    # Solve for the coefficients c using R
    try:
        c = np.linalg.solve(R, Q.T @ residuals_history[-1])
    except np.linalg.LinAlgError:
        c = np.zeros((m, 1))

    # Compute the new update
    update = theta_history[-1] - F @ c

    return update

# Compute loss function
def compute_loss(X, y, theta):
    m = len(y)
    predictions = sigmoid(X.dot(theta))
    loss = -1/m * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return loss

# Unaccelerated SGD implementation
def sgd(X, y, learning_rate, n_iterations, batch_size):
    theta = np.random.randn(X.shape[1], 1)
    loss_history = []

    for iteration in range(n_iterations):
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for i in range(0, len(X), batch_size):
            X_i = X_shuffled[i:i+batch_size]
            y_i = y_shuffled[i:i+batch_size]
            gradients = compute_gradient(X_i, y_i, theta)
            theta = theta - learning_rate * gradients
        loss = compute_loss(X, y, theta)
        loss_history.append(loss)
    
    return theta, loss_history

# SGD with Anderson Acceleration using normal equations
def sgd_anderson_norm(X, y, learning_rate, n_iterations, batch_size, m_history):
    theta = np.random.randn(X.shape[1], 1)
    loss_history = []
    theta_history = []
    residuals_history = []

    for iteration in range(n_iterations):
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for i in range(0, len(X), batch_size):
            X_i = X_shuffled[i:i+batch_size]
            y_i = y_shuffled[i:i+batch_size]
            gradients = compute_gradient(X_i, y_i, theta)
            theta = theta - learning_rate * gradients

            # Store the history
            if len(theta_history) >= m_history:
                theta_history.pop(0)
                residuals_history.pop(0)

            residual = gradients
            theta_history.append(theta.copy())
            residuals_history.append(residual.copy())

            # Apply Anderson Acceleration using normal equations
            if len(theta_history) > 1:
                theta = anderson_acceleration_norm(theta_history, residuals_history)

        loss = compute_loss(X, y, theta)
        loss_history.append(loss)

    return theta, loss_history

# SGD with Anderson Acceleration using QR
def sgd_anderson_qr(X, y, learning_rate, n_iterations, batch_size, m_history):
    theta = np.random.randn(X.shape[1], 1)
    loss_history = []
    theta_history = []
    residuals_history = []

    for iteration in range(n_iterations):
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for i in range(0, len(X), batch_size):
            X_i = X_shuffled[i:i+batch_size]
            y_i = y_shuffled[i:i+batch_size]
            gradients = compute_gradient(X_i, y_i, theta)
            theta = theta - learning_rate * gradients

            # Store the history
            if len(theta_history) >= m_history:
                theta_history.pop(0)
                residuals_history.pop(0)

            residual = gradients
            theta_history.append(theta.copy())
            residuals_history.append(residual.copy())

            # Apply Anderson Acceleration using QR
            if len(theta_history) > 1:
                theta = anderson_acceleration_qr(theta_history, residuals_history)

        loss = compute_loss(X, y, theta)
        loss_history.append(loss)

    return theta, loss_history

# Run the comparison
start_time = time.time()
theta_sgd, loss_history_sgd = sgd(X_b, y, learning_rate, n_iterations, batch_size)
time_sgd = time.time() - start_time

start_time = time.time()
theta_anderson_norm, loss_history_anderson_norm = sgd_anderson_norm(X_b, y, learning_rate, n_iterations, batch_size, m_history)
time_anderson_norm = time.time() - start_time

start_time = time.time()
theta_anderson_qr, loss_history_anderson_qr = sgd_anderson_qr(X_b, y, learning_rate, n_iterations, batch_size, m_history)
time_anderson_qr = time.time() - start_time

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(range(n_iterations), loss_history_sgd, label='SGD')
plt.plot(range(n_iterations), loss_history_anderson_norm, label='SGD with Anderson Acceleration (Normal Equations)')
plt.plot(range(n_iterations), loss_history_anderson_qr, label='SGD with Anderson Acceleration (QR)')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs Iterations for Logistic Regression')
plt.legend()
plt.grid(True)
plt.show()

# Print execution times and final parameters
print(f"Unaccelerated SGD Time: {time_sgd:.4f} seconds")
print(f"SGD with Anderson Acceleration (Normal Equations) Time: {time_anderson_norm:.4f} seconds")
print(f"SGD with Anderson Acceleration (QR) Time: {time_anderson_qr:.4f} seconds")

print(f"Unaccelerated SGD Final Theta: {theta_sgd.ravel()}")
print(f"SGD with Anderson Acceleration (Normal Equations) Final Theta: {theta_anderson_norm.ravel()}")
print(f"SGD with Anderson Acceleration (QR) Final Theta: {theta_anderson_qr.ravel()}")