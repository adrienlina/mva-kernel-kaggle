import re

import numpy as np
import pandas as pd


def get_words(k):
    words = ['']
    for i in range(k):
        new_words = []
        for letter in 'AGCT':
            for word in words:
                new_words.append(letter + word)
        words = new_words

    return words

def get_index(string, words):
    return np.asarray([
        len(re.findall('(?=%s)' % word, string))
        for word in words
    ])

def index_seq(seq, words):
    n = seq.shape[0]
    d = len(words)

    X = np.zeros((n,d))
    for i in range(n):
        if i % 100 == 0:
            print(i)
        X[i,:] = get_index(seq[i,0], words)

    return X

def transform_to_index_and_save(k):
    words = get_words(k)
    for index in range(3):
        seq_train = pd.read_csv('Xtr%s.csv' % index, names='1').values
        X_train = index_seq(seq_train, words)
        np.savetxt('spectral_preindexed/Xtr%s_spectral_%s.gz' % (index, k), X_train)

        seq_test = pd.read_csv('Xte%s.csv' % index, names='1').values
        X_test = index_seq(seq_test, words)
        np.savetxt('spectral_preindexed/Xte%s_spectral_%s.gz' % (index, k), X_test)


def load_data_k_spectrum(k):
    data = {}
    for index in range(3):
        X_train = np.loadtxt('spectral_preindexed/Xtr%s_spectral_%s.gz' % (index, k))
        seq_train = pd.read_csv('Xtr%s.csv' % index, names='1').values
        X_test = np.loadtxt('spectral_preindexed/Xte%s_spectral_%s.gz' % (index, k))
        seq_test = pd.read_csv('Xte%s.csv' % index, names='1').values
        labels = pd.read_csv('Ytr%s.csv' % index, names=('Id', 'Bound'), skiprows=[0], delimiter=',')

        X_train_mean = X_train.mean(axis=0)
        X_train_center = X_train - X_train_mean
        X_test_center = X_test - X_train_mean

        data[index] = {
            'seq_train': seq_train,
            'X_train': X_train,
            'y_train': labels['Bound'].values,
            'seq_test': seq_test,
            'X_test': X_test,
            'ids': labels['Id'].values,
        }

    return data
