import numpy as np


def compute_K_matrix(kernel, X):
    """
    Compute the matrix K where K(i,j) = kernel(X(i,:), X(j,:))

    PARAMETERS:
    - kernel: the kernel (vectorized) function to use
    - X (n,d): the data points to compute

    RETURNS:
    - Matrix of size (n,n)
    """
    return kernel(X.T, X.T)


def kernel_ridge_regression(kernel, X, y, reg):
    """
    Compute alpha for a ridge regression as explained in course slides p.90-92

    PARAMETERS:
    - kernel: the kernel (vectorized) function to use
    - X (n,d): train data points
    - y (n,): train labels (0 or 1)
    - reg: the regularization parameter

    RETURNS:
    - alpha of size (n,)
    """
    n = X.shape[0]
    K = compute_K_matrix(kernel, X)

    return np.linalg.inv(K + reg * np.eye(n)) @ y


def get_ridge_prediction(kernel, X_train, X_test, alpha):
    """
    Get ridge regression prediction

    PARAMETERS:
    - kernel: the kernel (vectorized) function to use
    - X_train (n,d): the data points that was used to train the ridge regression
    - X_test (m,d): the data points for which we want a prediction
    - alpha (n,): the coefficient for each point, as explained in the course
        slides p.90

    RETURNS:
    - (0-1) array of size (m,)
    """
    K_x = kernel(X_train.T, X_test.T)
    proba = alpha @ K_x
    return (proba >= 1/2).astype(int)
