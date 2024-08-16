import numpy as np

# Generate synthetic data for linear regression
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term
X_b = np.c_[np.ones((100, 1)), X]

# Hyperparameters
learning_rate = 0.01
n_iterations = 50
batch_size = 20
m_history = 5  # Anderson Acceleration history length

# Initialize weights
theta = np.random.randn(2, 1)

# Compute gradient function
def compute_gradient(X, y, theta):
    m = len(y)
    gradients = 2/m * X.T.dot(X.dot(theta) - y)
    return gradients

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

# Function to compute loss
def compute_loss(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    loss = (1/m) * np.sum((predictions - y) ** 2)
    return loss

# SGD with Anderson Acceleration using QR
theta_history = []
residuals_history = []
loss_history_aa_qr = []

for iteration in range(n_iterations):
    indices = np.random.permutation(len(X_b))
    X_b_shuffled = X_b[indices]
    y_shuffled = y[indices]
    for i in range(0, len(X_b), batch_size):
        X_i = X_b_shuffled[i:i+batch_size]
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
    
    loss = compute_loss(X_b, y, theta)
    loss_history_aa_qr.append(loss)

print("SGD with Anderson Acceleration using QR Theta:", theta)
print("SGD with Anderson Acceleration using QR Loss History:", loss_history_aa_qr)