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

# Initialize weights
theta = np.random.randn(2, 1)

# Compute gradient function
def compute_gradient(X, y, theta):
    m = len(y)
    gradients = 2/m * X.T.dot(X.dot(theta) - y)
    return gradients

# Function to compute loss
def compute_loss(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    loss = (1/m) * np.sum((predictions - y) ** 2)
    return loss

# SGD implementation
loss_history_sgd = []
for iteration in range(n_iterations):
    indices = np.random.permutation(len(X_b))
    X_b_shuffled = X_b[indices]
    y_shuffled = y[indices]
    for i in range(0, len(X_b), batch_size):
        X_i = X_b_shuffled[i:i+batch_size]
        y_i = y_shuffled[i:i+batch_size]
        gradients = compute_gradient(X_i, y_i, theta)
        theta = theta - learning_rate * gradients
    loss = compute_loss(X_b, y, theta)
    loss_history_sgd.append(loss)

print("SGD Theta:", theta)
print("SGD Loss History:", loss_history_sgd)