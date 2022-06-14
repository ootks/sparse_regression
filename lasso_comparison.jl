#using Random, Distributions, MLJLinearModels
#using StatsBase: sample
include("subset_selection/hyperbolic_relaxation.jl")
#n = 100
#m = 30
#noise = 0.1
#k = 6
#A = rand(Normal(), m, n)
#planted = sample([i for i=1:n], k, replace=false)
#println(planted)
#b = sum([A[:,planted[l]] for l in 1:k]) + noise * rand(m)
#
##Find LASSO subset
#i = 0
#theta = []
#S_lasso = []
#for i in 1:100
#    lasso = LassoRegression(lambda = 0.01*i)
#    global theta = MLJLinearModels.fit(lasso, A, b)
#    global S_lasso = [abs(theta[i]) > 0.001 for i=1:n]
#    if sum(S_lasso) <= k
#        selection = []
#        for i in 1:n
#            if S_lasso[i]
#                push!(selection, i)
#            end
#        end
#        println(selection)
#        break
#    end
#    i += 1
#    if i > 100
#        println("Failed")
#        break
#    end
#end
#
##println(theta)
#println(linear_regression_objective(A[:, S_lasso], b))
#
## Find subset using heuristic
#x = find_subset(A, b, k)
#println(x)
#println(linear_regression_objective(A[:, [i in x for i=1:n]], b))

#using CSV, DataFrames, LinearAlgebra
#using MLJLinearModels
#include("hyperbolic_relaxation.jl")
#df =  CSV.File("/home/kshu/Downloads/Life Expectancy Data.csv", drop=["Country", "Status", "Year"], types=Float64) |> DataFrame
#df = df[completecases(df), :]
#b = Vector{Float64}(df[!, "Life expectancy "])
#A = Matrix{Float64}(df[!, Not("Life expectancy ")])
#n = size(A, 2)
#
## Normalize data
#C = transpose(A) * A
#D = Diagonal([1/sqrt(C[i,i]) for i=1:n])
#A = A * D
#
##display(b)
##display(A)
#
##println(size(A, 2))
##for k in 1:n-5
##    println(k)
##    x = find_subset(A, b, k)
##    println(x)
##    println(linear_regression_objective(A[:, [i in x for i=1:n]], b))
##end
#
#selections = Vector{Any}([nothing for i in 1:n])
#vals = Vector{Any}([nothing for i in 1:n])
#for i in 0:100
#    lasso = LassoRegression(lambda = 0.01*i)
#    theta = MLJLinearModels.fit(lasso, A, b)
#    S_lasso = [abs(theta[j]) > 0.001 for j=1:n]
#    k = sum(S_lasso)
#    if k == 0
#        continue
#    end
#    selection = []
#    for i in 1:n
#        if S_lasso[i]
#            push!(selection, i)
#        end
#    end
#    val = linear_regression_objective(A[:, S_lasso], b)
#    if vals[k] == nothing || val < vals[k]
#        vals[k] = val
#        selections[k] = selection
#    end
#end
#for k in 1:n
#    println(k)
#    println(selections[k])
#    println(vals[k])
#end

using Distributions
#n = 100
#m = 50
#A = rand(Normal(), m,n)
#b = sum([A[:, k] for k=2:6])
#A[:, 1] = b
#
#using DelimitedFiles
#writedlm("gaussian_matrix.csv", A, ',')
#
using CSV, DataFrames, LinearAlgebra
function proj_away(b, x)
    return b - dot(b, x)/dot(x, x) * x
end
#df =  CSV.File("gaussian_matrix.csv", types=Float64) |> DataFrame
#df = df[completecases(df), :]
#b = Vector{Float64}(df[!, 1])
#A = Matrix{Float64}(df[!, 2:end])
n = 200
m = 50
A = rand(Normal(), m,n)
b = sum([A[:, k] for k=2:6])
x = find_subset(A, b, 5)
println(x)
println(linear_regression_objective(A[:, [i in x for i=1:n]], b))

vecs = []
for i=1:5
    max_dot = 0
    best = 0
    for j=1:n
        dot_prod = abs(dot(b, A[:, j]))
        if dot_prod > max_dot
            max_dot = dot_prod
            best = j
        end
    end
    push!(vecs, best)
    v = A[:, best]
    global b = proj_away(b, v)
    for j=1:n
        global A[:, j] = proj_away(A[:, j], v)
    end
end
println(b)
println(vecs)
