import numpy as np
import matplotlib.pyplot as plt

from data_manipulation import split_train_test_valid, get_precision
from regressions import get_ridge_prediction, get_best_reg_param


def produce_results(preds, ids):
    """
    Save resuts to a submission for Kaggle (submission.csv)
    """
    data = [
        '%s,%s' % (id, pred) for pred, id in zip(preds, ids)
    ]

    data.insert(0, 'Id,Bound')
    with open('submission.csv', 'w') as f:
        f.write('\n'.join(data))

def train_tune_and_pred(kernel, method, DATA):
    """
    Train, tune regularization and predict for the hidden test data points using
    a kernel and a method
    """
    preds = []
    ids = []
    for index in range(3):
        train_X, train_y, test_X, test_y, valid_X, valid_y = split_train_test_valid(
            DATA[index]['X_train'],
            DATA[index]['y_train']
        )

        # Finding best parameter
        reg_param, _ = get_best_reg_param(kernel, method, (train_X, train_y, test_X, test_y))

        # Validation precision
        K = kernel(train_X, train_X)
        K_x = kernel(train_X, valid_X)
        alpha = method(K, train_y, reg_param)
        pred = get_ridge_prediction(K_x, alpha)
        print("Dataset %s has found a parameter (%s) with validation precision %.3f" % (index, reg_param, get_precision(pred, valid_y)))

        continue
        # Kaggle submission
        K = kernel(DATA[index]['X_train'], DATA[index]['X_train'])
        K_x = kernel(DATA[index]['X_train'], DATA[index]['X_test'])
        alpha = method(K, DATA[index]['y_train'], reg_param)
        pred = get_ridge_prediction(K_x, alpha)

        preds += list(pred)

    produce_results(preds, range(3000))


def see_variability(kernel, method, DATA, reg_params, N=10):
    """
    Compute over N iterations the validation performance of a kernel and a method
    """
    results = np.zeros((3,N))
    for index in range(3):
        print(index)
        for n in range(N):
            train_X, train_y, test_X, test_y, valid_X, valid_y = split_train_test_valid(
                DATA[index]['X_train'],
                DATA[index]['y_train']
            )

            K = kernel(train_X, train_X)
            alpha = method(K, train_y, reg_params[index])

            # Validation precision
            K_x = kernel(train_X, valid_X)
            pred = get_ridge_prediction(K_x, alpha)
            results[index,n] = get_precision(pred, valid_y)

    print(results.mean(axis=1))

    return results
