## Samplers using Distributions package

using Distributions

# single sample
categorical_sample(p::Vector{Float64}) = categorical_sample(p, 1)[:,1]
poisson_sample(lambda::Float64) = poisson_sample(lambda, 1)[1]
gamma_sample(a::Float64, b::Float64) = gamma_sample(a, b, 1)[1]
dirichlet_sample(alpha::Vector{Float64}) = dirichlet_sample(alpha, 1)[:,1]

# samplers
function categorical_sample(p::Vector{Float64}, N::Int)
    K = length(p)
    Z = zeros(K, N)
    Z_tmp = rand(Categorical(p), N)
    for k in 1 : K
        Z[k,find(Z_tmp.==k)] = 1
    end
    return Z
end

function poisson_sample(lambda::Float64, N::Int)
    X = rand(Poisson(lambda), N)
    return X
end

function gamma_sample(a::Float64, b::Float64, N::Int)
    # a: shape parameter
    # b: rate parameter
    lambda = rand(Gamma(a, 1 / b), N)
    return lambda
end

function dirichlet_sample(alpha::Vector{Float64}, N::Int)
    phi = zeros(length(alpha), N)
    for n in 1 : N
        phi[:,n] = rand(Dirichlet(alpha))
    end
    return phi 
end
