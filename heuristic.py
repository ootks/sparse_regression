import numpy as np

def char_coeff_FL(X, k):
    if k == 0:
        return 1
    elif k == 1:
        return np.trace(X)
    powers = [X]
    half_k = int(k / 2) 
    for i in range(half_k):
        powers.append(powers[-1] @ X)
    traces = [np.trace(A) for A in powers]
    for i in range(half_k, k):
        traces.append(sum(sum(powers[-1] * powers[i-half_k])))
    x = np.array([[fl_matrix_entry(i, j, traces) for j in range(k)] for i in range(k)])
    return np.linalg.det(x)

def fl_matrix_entry(i, j, l):
    if i > j + 1:
        return 0
    elif i == j + 1:
        return len(l) - i - 1
    else:
        return l[j-i]

def swap(X, i, j):
    for k in range(len(X)):
        X[k,i], X[k,j]  = X[k,j], X[k,i]
    for k in range(len(X)):
        X[i,k], X[j,k]  = X[j,k], X[i,k]

def conditional_char(X, t, k):
    schur = X[t:, t:] - X[t:, :t] @ np.linalg.inv(X[:t, :t]) @ X[:t, t:]
    return np.linalg.det(X[:t, :t]) * char_coeff_FL(schur, k-t)

def find_subset(A, b, k):
    n = A.shape[1]
    T = []
    X = np.transpose(A)@A
    Z = X + (np.transpose(A) @ np.outer(b, b) @ A)

    for t in range(k):
        best = -1
        best_heur = 0
        for j in range(t, n):
            if j != t:
                swap(X, t, j)
                swap(Z, t, j)
            
            pX = conditional_char(X, t+1, k)
            pZ = conditional_char(Z, t+1, k)
            heur = pZ / pX
            if heur > best_heur:
                best = j
                best_heur = heur
            swap(X, t, j)
            swap(Z, t, j)
        try:
            while True:
                best = T.index(best)
        except ValueError:
            T.append(best)
        swap(X, t, best)
        swap(Z, t, best)
    return T

def lin_reg(A, b):
    temp = np.transpose(A) @ b
    return np.dot(b, b) - np.dot(temp, np.inv(np.transpose(A) @ A) @ temp)