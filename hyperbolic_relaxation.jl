using LinearAlgebra
using Profile
"""
    char_coeffs(X, k)

    Compute the degree k characteristic coefficients of the matrix X.
    uses Faddeev LeVerrier method.

    Faster for n = 1000, k = 10
"""
function char_coeff_FL(X, k::Int64)
    powers = [X]
    half_k = (k÷2)
    for i=1:half_k
        push!(powers, Symmetric(last(powers) * X))
    end

    traces = [tr(A) for A in powers]
    for i=(half_k+1):k
        push!(traces, dot(last(powers), powers[i-half_k]))
    end

    det([fl_matrix_entry(i, j, traces) for i=1:k, j=1:k])
end

function fl_matrix_entry(i::Int64, j::Int64, l::Vector)
    if i > j + 1
        0
    elseif i == j + 1
        length(l) - i
    else
        l[j-i+1]
    end
end

function swap!(X::AbstractMatrix, i::Integer, j::Integer)
    for k = 1:size(X,1)
        X[k,i], X[k,j] = X[k,j], X[k,i]
    end
    for k = 1:size(X,1)
        X[i,k], X[j,k] = X[j,k], X[i,k]
    end
end

function conditional_char(X, t, k, n)
    X[t,t] * char_coeff_FL(X[t+1:n,t+1:n] - X[t+1:n, t:t]*X[t:t, t+1:n]/X[t,t], k-t)
end

"""
Diagonalize the matrix, and then compute the elementary symmetric polynomials of the matrix
"""
function char_coeff_eigen(X, k::Int64)
    e = eigvals(Symmetric(X))
    n = length(e)
    # Dynamic programming table. 
    # e_k^n = e_k^n + xn e_{k-1}^{n-1}
    dp_table = zeros(n+1, k)
    for j=2:n+1
        dp_table[j, 1] = dp_table[j-1, 1] + e[j-1]
    end
    for i=2:k
        for j=2:n+1
            dp_table[j,i] = dp_table[j-1, i] + e[j-1] * dp_table[j-1,i-1]
        end
    end
    dp_table[n+1,k]
end

"""
Heuristic for finding a set of k variables T that minimizes the least squares
error:
min |A|_T x - b|^2

A: An n x m matrix
b: A vector of length n
k: Size of output set
search: If this is true, find the best element to add to T in each round.
If false, find the first element that does at least as well as char_coeff_FL(X)
verbose: If this is true, print out the scores over time.
"""
function find_subset(A::Matrix{Float64}, b::Array{Float64}, k::Int64, 
                    search::Bool=true, verbose::Bool=false)
    n = size(A,2)
    # Features selected
    T = [] 
    # Maintain X = A^T A \ T, where \ denotes the schur complement
    X = transpose(A)*A
    # Maintain Z = (A^T A + A^Tbb^TA) \ T
    Z = X + (transpose(A)*b*transpose(b)*A)

    if !search
        pX = char_coeff_FL(X, k)
        pZ = char_coeff_FL(Z, k)
        best_char = (pZ-pX)/pX
    end
    for t=1:k
        if verbose
            println("Round ", t)
        end
        best = 0
        if search
            best_char = 0
        end

        for j=t:n
            # Reposition so that the candidate is in position t.
            if j != t
                swap!(X, t, j)
                swap!(Z, t, j)
            end
            pX = conditional_char(X, t, k, n)
            pZ = conditional_char(Z, t, k, n)
            char = (pZ-pX)/pX
            if char > best_char
                best = j
                if search
                    best_char = char
                else
                    break
                end
            end
            swap!(X, t, j)
            swap!(Z, t, j)
        end
        # Add in the index of the best entry.
        index = indexin(best, T)[1]
        while index != nothing
            best = index
            index = indexin(best, T)[1]
        end
        push!(T, best)
        if search
            swap!(X, t, best)
            swap!(Z, t, best)
        end
        X[t+1:n, t+1:n] -= X[t+1:n, t:t]*X[t:t, t+1:n]/X[t,t]
        Z[t+1:n, t+1:n] -= Z[t+1:n, t:t]*Z[t:t, t+1:n]/Z[t,t]
    end

    return T
end
#@time (for i=1:100; find_subset(rand(10000, 100), rand(10000), 10, false); end)
@time (for i=1:100; find_subset(rand(10000, 100), rand(10000), 10, true); end)
