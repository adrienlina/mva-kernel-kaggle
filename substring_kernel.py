def momoize(func):
    memory = {}
    def momoized(*args):
        key = '-'.join('[%s]' % arg for arg in args)
        if key not in memory:
            memory[key] = func(*args)

        return memory[key]

    return momoized


@momoize
def B_k(LAMBDA, k, str1, str2):
    if k == 0:
        return 1

    len_1 = len(str1)
    len_2 = len(str2)
    if len_1 < k or len_2 < k:
        return 0

    sub_1 = str1[0:-1]
    sub_2 = str2[0:-1]

    return (LAMBDA*B_k(LAMBDA, k, sub_1, str2)
            + LAMBDA*B_k(LAMBDA, k, str1, sub_2)
            - LAMBDA**2*B_k(LAMBDA, k, sub_1, sub_2)
            + (LAMBDA**2*B_k(LAMBDA, k-1, sub_1, sub_2) if str1[-1] == str2[-1] else 0)
           )


@momoize
def K_k(LAMBDA, k, str1, str2):
    if k == 0:
        return 1

    len_1 = len(str1)
    len_2 = len(str2)
    if len_1 < k or len_2 < k:
        return 0

    sub_1 = str1[0:-1]
    a = str1[-1]

    return (K_k(LAMBDA, k, sub_1, str2)
            + LAMBDA**2 * sum(B_k(LAMBDA, k-1, sub_1, str2[0:j]) for j in range(len_2) if str2[j] == a)
           )


def substring_kernel(str1, str2):
    n1 = str1.shape[0]
    n2 = str2.shape[0]

    K = np.zeros((n1,n2))
    for i in range(n1):
        print(i)
        for j in range(n2):
            K[i,j] = K_k(0.1, 2, str1[i,0], str2[j,0])

    return K
