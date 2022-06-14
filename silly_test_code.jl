include("subset_selection/hyperbolic_relaxation.jl")

k = 5
function entries(i, j)::Float64
    if i == j
        1
    elseif i <= k
        if j <= k 
            -1/(2*k-1)
        else 
            1/(2*k-1)
        end
    else
        0
    end
end
X = [entries(i,j) for i=1:2*k, j=1:2*k]
display(Symmetric(X))
println()
b = X * [i <= k ? 1. : 0. for i =1:2*k]
display(b)
println()

Z = X + b * transpose(b)
display(Z)
println()

println("Z Cond char", conditional_char(Symmetric(Z), 1, k, 2*k))
println("X Cond char", conditional_char(Symmetric(X), 1, k, 2*k))
println("Score, ", conditional_char(Symmetric(Z), 1, k, 2*k) / conditional_char(Symmetric(X), 1, k, 2*k))

swap!(X, 1, k+1)
swap!(Z, 1, k+1)

println("Z Cond char", conditional_char(Symmetric(Z), 1, k, 2*k))
println("X Cond char", conditional_char(Symmetric(X), 1, k, 2*k))
println("Score, ", conditional_char(Symmetric(Z), 1, k, 2*k) / conditional_char(Symmetric(X), 1, k, 2*k))
