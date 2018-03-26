import numpy as np


TRAIN_RATIO = 0.6
TEST_RATIO = 0.2
VALID_RATIO = 0.2


def split_train_test_valid(X, y):
    """
    Split a dataset in 3 sub-datasets

    PARAMETERS:
    - X (n,d): train data points to split
    - y (n,): train labels to split

    RETURNS:
    - 6-tuple of sub-datasets:
        (
            X_train (n1,d),
            y_train (n1,),
            X_test (n2,d),
            y_test (n2,),
            X_valid (n3, d),
            y_valid (n3,)
        )
    """
    n_items = y.shape[0]
    random_order = np.random.permutation(range(n_items))

    train_index = round(n_items * TRAIN_RATIO)
    test_index = train_index + round(n_items * TEST_RATIO)
    valid_index = test_index + round(n_items * VALID_RATIO)

    X_train = X[random_order[0:train_index], :]
    y_train = y[random_order[0:train_index]]
    X_test = X[random_order[train_index:test_index], :]
    y_test = y[random_order[train_index:test_index]]
    X_valid = X[random_order[test_index:valid_index], :]
    y_valid = y[random_order[test_index:valid_index]]

    return X_train, y_train, X_test, y_test, X_valid, y_valid


def get_precision(y_pred, y_true):
    """
    Returns the ratio of correct labels given a prediction and a true label

    PARAMETERS:
    - y_pred (n,): Predicted labels
    - y_true (n,): True labels

    RETURNS:
    - float
    """
    return np.sum(y_pred == y_true) / len(y_pred)
