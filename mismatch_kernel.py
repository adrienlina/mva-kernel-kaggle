import numpy as np
import Levenshtein

from spectrum_kernel import get_words


def get_mismatch_matrix(k,m):
    """
    Compute the mismatch mixing matrix for A(k) up to m mismatches

    PARAMETERS:
    - k: the length of strings in the alphabet
    - m: the maximum number of mismatches allowed

    RETURNS:
    - (4^k, 4^k) mixing matrix
    """
    words = get_words(k)
    N = len(words)

    mismatch_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            if Levenshtein.hamming(words[i], words[j]) <= m:
                mismatch_matrix[i,j] = 1/2
                mismatch_matrix[j,i] = 1/2

    return mismatch_matrix


def get_exp_mismatch_matrix(k, _lambda):
    """
    Compute the mismatch mixing matrix for A(k) with an _lambda-exponentially
    decaying mixing coefficient in the Hamming distance (number of character
    mismatches).

    Eg: 'AAAA' and 'AABB' have a Hamming distance of 2, thus a mixing coefficient
        of _lambda**2

    PARAMETERS:
    - k: the length of strings in the alphabet
    - _lambda: the exponential parameter of the decay per mismatches

    RETURNS:
    - (4^k, 4^k) mixing matrix
    """

    words = get_words(k)
    N = len(words)

    exp_mismatch_matrix = np.zeros((N, N))
    for i in range(N):
        exp_mismatch_matrix[i,i] = 1
        for j in range(i+1, N):
            exp_mismatch_matrix[i,j] = _lambda**Levenshtein.hamming(words[i], words[j])
            exp_mismatch_matrix[j,i] = exp_mismatch_matrix[i,j]

    return exp_mismatch_matrix
