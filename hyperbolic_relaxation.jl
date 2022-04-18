using LinearAlgebra
"""
    char_coeffs(X, k)

    Compute the degree k characteristic coefficients of the matrix X.
    uses Faddeev LeVerrier method.

    Faster for n = 1000, k = 10
"""
function char_coeff_FL(X, k::Int64)
    powers = [X]
    half_k = (k÷2)
    @inbounds for i=1:half_k
        push!(powers, Symmetric(last(powers) * X))
    end

    traces = [tr(A) for A in powers]
    @inbounds for i=(half_k+1):k
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
    det(X[1:t,1:t]) * char_coeff_FL(X[t+1:n,t+1:n] - X[t+1:n, 1:t]*inv(X[1:t,1:t])*X[1:t, t+1:n], k-t)
end

function find_subset(A, b, k)
    X = transpose(A)*A
    Z = X + (transpose(A)*b*transpose(b)*A)
    
    T = [] 
    n = size(X,1)

    for t=1:k
        best = 0
        best_char = 0
        println("Round ", t)
        for j=t:n
            # Reposition so that the candidate is in position t.
            if j != t
                swap!(X, t, j)
                swap!(Z, t, j)
            end
            pX = conditional_char(X, t, k, n)
            pZ = conditional_char(Z, t, k, n)
            char = (pZ-pX)/pX
            if char > best
                best = j
                best_char = char
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
        swap!(X, t, best)
        swap!(Z, t, best)
    end

    return T
end

println(find_subset(Symmetric(Matrix(I, 1000, 1000)), [1 for i=1:1000], 5))

"""
Diagonalize the matrix, and then compute the elementary symmetric polynomials of the matrix
"""
function char_coeff_eigen(X, k::Int64)
    e = eigvals(Symmetric(X))
    n = length(e)
    # Dynamic programming table. 
    # e_k^n = e_k^n + xn e_{k-1}^{n-1}
    dp_table = zeros(n+1, k)
    @inbounds for j=2:n+1
        dp_table[j, 1] = dp_table[j-1, 1] + e[j-1]
    end
    @inbounds for i=2:k
        @inbounds for j=2:n+1
            dp_table[j,i] = dp_table[j-1, i] + e[j-1] * dp_table[j-1,i-1]
        end
    end
    dp_table[n+1,k]
end

##X = Symmetric(Matrix(I, 10000, 10000))
## Warm up JIT
#X1 = Symmetric(rand(Float64, (2000,2000)))
#
#println(char_coeff_eigen(X1, 10))
#println(char_coeff_FL(X1, 10))
#
#t1 = 0
#t2 = 0
#for i=1:100
#    X = Symmetric(rand(Float64, (1000,1000)))
#    #X = Symmetric(Matrix(I, 1000, 1000))
#
#    stat = @timed println(char_coeff_eigen(X, 10))
#    global t1 += stat.time
#    stat = @timed println(char_coeff_FL(X, 10)/factorial(10))
#    global t2 += stat.time
#end
#println(t1)
#println(t2)
