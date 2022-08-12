import numpy as np
import math

def esp(x, k):
    n = len(x)
    if k == 0:
        return 1
    if k == 1:
        return sum(x)
    S = np.zeros((n+1, k))
    for j in range(1, n+1):
        S[j, 0] = S[j-1, 0] + x[j-1]
    for i in range(1, k):
        for j in range(1, n+1):
            S[j, i] = S[j-1, i] + x[j-1] * S[j-1, i-1]
    return S[n, k-1]

def char_coeff_eigen(X, k):
    return esp(np.linalg.eigvalsh(X), k)

def char_coeff(X, k):
    if k == 0:
        return 1
    #if k < 8:
    #return char_coeff_FL(X, k)
    return char_coeff_eigen(X, k)
    #return char_poly_built(X, k)
    
def maximal_root(p):
    return max(np.roots(p))

def newton_method(p):
    # start at a point that is larger than the maximum root of p
    x = 100
    d = len(p)-1
    dpdx = [(d - i) * p[i] for i in range(d)]
    
    iters = 0
    while abs(evaluate(p, x)) > 1e-3:
        x -= evaluate(p, x) / evaluate(dpdx, x)
        iters += 1
        if iters > 100000:
            print("failed")
            print(p)
            print(np.roots(p))
            break
    return x

def evaluate(p, x):
    # Horn evaluation
    val = 0
    for c in p:
        val *= x
        val += c
    return val

def swap(X, i, j):
    if i == j:
        return
    for k in range(len(X)):
        X[k,i], X[k,j]  = X[k,j], X[k,i]
    for k in range(len(X)):
        X[i,k], X[j,k]  = X[j,k], X[i,k]

def conditional_char(X, t, k):
    schur = X[t:, t:] - X[t:, :t] @ np.linalg.inv(X[:t, :t]) @ X[:t, t:]
    return np.linalg.det(X[:t, :t]) * char_coeff(schur, k-t)

def root_heur(X, t, k, D = None):
    if D is None:
        D = np.eye(len(X))
    # Evaluate p(X + t I) in k places
    xs = [0.1*x for x in range(k+1)]
    vals = [conditional_char(x * D - X, t+1, k) for x in xs]
    # Find the coefficients of p(X+tI)
    p = np.polyfit(xs, vals, k)
    
    # Use Newtons' method to find maximal root
    return newton_method(p)

def find_subset(A, k):
    n = A.shape[1]
    T = []
    X = A.copy()
    #Normalize X so that its eigenvalues lie in the range [1/2, 1]
    eigs = np.linalg.eigvalsh(X)
    max_eig = max(eigs)
    min_eig = min(eigs)
    scal = 1/(2.1*(max_eig - min_eig))
    X = scal * X + (0.5 - min_eig * scal) * np.eye(n)
    
    bests = []
    for t in range(k):
        best = -1
        best_heur = 0
        print("Round ", t)
        for j in range(t, n):
            swap(X, t, j)
            heur = root_heur(X, t, k)
            if heur > best_heur:
                best = j
                best_heur = heur
            swap(X, t, j)
        swap(X, t, best)
        try:
            while True:
                best = T.index(best)
        except ValueError:
            print("Best Found ", best)
            T.append(best)
        bests.append(best_heur - 1)
    return T, bests

import random
# Produce Erdos Renyi Random Graph
p = 0.2
n = 500
k = 10
S = list(range(k))
X = np.zeros((n,n))
for i in range(n):
    for j in range(i, n):
        if i == j:
            X[i,i] = 0
        elif i in S and j in S:
            X[i,j] = X[j,i] = 1 - p
        else:
            X[i,j] = X[j,i] = np.random.binomial(1,p) - p
print("True set ", S)
T,_ = find_subset(X, k)
print("Found set ", T)
print([[X[i,j] for i in T] for j in T])
