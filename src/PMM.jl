###################################
## Bayesian Poisson Mixture Model

import StatsFuns.logsumexp
include("sampler.jl")

####################
## Types
type Gam
    # Parameters of Gamma distribution
    a::Vector{Float64}
    b::Float64
end

type BPMM
    # Parameters of Bayesian Poisson Mixture Model 
    D::Int
    K::Int
    alpha::Vector{Float64}
    cmp::Vector{Gam}
end

type Poi
    # Parameters of Poisson Distribution
    lambda::Vector{Float64}
end

type PMM
    # Parameters of Poisson Mixture Model
    D::Int
    K::Int
    phi::Vector{Float64}
    cmp::Vector{Poi}
end

####################
## Common functions
function sample_PMM(bpmm::BPMM)
    cmp = Vector{Poi}()
    for c in bpmm.cmp
        lambda = Vector{Float64}()
        for d in 1 : bpmm.D
            push!(lambda, gamma_sample(c.a[d], c.b))
        end
        push!(cmp, Poi(lambda))  
    end
    phi = dirichlet_sample(bpmm.alpha)
    return PMM(bpmm.D, bpmm.K, phi, cmp)
end

function sample_data(pmm::PMM, N::Int)
    X = zeros(pmm.D, N)
    Z = categorical_sample(pmm.phi, N)
    for n in 1 : N
        k = indmax(Z[:, n])
        for d in 1 : pmm.D
            X[d,n] = poisson_sample(pmm.cmp[k].lambda[d])
        end
    end
    return X, Z
end

function init_Z(X::Matrix{Float64}, bpmm::BPMM)
    N = size(X, 2)
    K = bpmm.K
    Z = categorical_sample(ones(K)/K, N)    
    return Z
end

function calc_bound(X::Matrix{Float64}, pri::BPMM, pos::BPMM)
    ln_expt_Z = update_Z(pos, X)
    expt_Z = exp(ln_expt_Z)
    K, N = size(expt_Z)
    D = size(X, 1)

    expt_ln_lambda = zeros(D, K)
    expt_lambda = zeros(D, K)
    expt_lik = 0
    for k in 1 : K
        expt_ln_lambda[:,k] = digamma(pos.cmp[k].a) - log(pos.cmp[k].b)
        expt_lambda[:,k] = pos.cmp[k].a / pos.cmp[k].b
        for n in 1 : N
            expt_lik += expt_Z[k,n] * (X[:, n]' * expt_ln_lambda[:,k]
                                       - sum(expt_lambda[:,k]) - sum(lgamma(X[:,n]+1)))[1]
        end
    end
    
    expt_ln_pZ = sum(expt_Z' * (digamma(pos.alpha) - digamma(sum(pos.alpha))))
    expt_ln_qZ = sum(expt_Z .* ln_expt_Z)
    
    KL_lambda = 0
    for k in 1 : K
        KL_lambda += (sum(pos.cmp[k].a)*log(pos.cmp[k].b) - sum(pri.cmp[k].a)*log(pri.cmp[k].b)
                      - sum(lgamma(pos.cmp[k].a)) + sum(lgamma(pri.cmp[k].a))
                      + (pos.cmp[k].a - pri.cmp[k].a)' * expt_ln_lambda[:,k]
                      + (pri.cmp[k].b - pos.cmp[k].b) * sum(expt_lambda[:,k])
                      )[1]
    end
    KL_pi = (lgamma(sum(pos.alpha)) - lgamma(sum(pri.alpha))
             - sum(lgamma(pos.alpha)) + sum(lgamma(pri.alpha))
             + (pos.alpha - pri.alpha)' * (digamma(pos.alpha) - digamma(sum(pos.alpha)))
             )[1]
    
    VB = expt_lik + expt_ln_pZ - expt_ln_qZ - (KL_lambda + KL_pi)
    return VB
end

function add_stats(bpmm::BPMM, X::Matrix{Float64}, Z::Matrix{Float64})
    D = bpmm.D
    K = bpmm.K
    sum_Z = sum(Z, 2)
    alpha = [bpmm.alpha[k] + sum_Z[k] for k in 1 : K]
    cmp = Vector{Gam}()

    XZ = X*Z';
    for k in 1 : K
        a = [(bpmm.cmp[k].a[d] + XZ[d,k])::Float64 for d in 1 : D]
        b = bpmm.cmp[k].b + sum_Z[k]
        push!(cmp, Gam(a, b))
    end
    return BPMM(D, K, alpha, cmp)
end

remove_stats(bpmm::BPMM, X::Matrix{Float64}, Z::Matrix{Float64}) = add_stats(bpmm, X, -Z)

####################
## used for Variational Inference
function update_Z(bpmm::BPMM, X::Matrix{Float64})
    D, N = size(X)
    K = bpmm.K
    ln_expt_Z = zeros(K, N)
    tmp = zeros(K)

    sum_digamma_tmp = digamma(sum(bpmm.alpha))
    for k in 1 : K
        tmp[k] = - sum(bpmm.cmp[k].a) / bpmm.cmp[k].b
        tmp[k] += digamma(bpmm.alpha[k]) - sum_digamma_tmp
    end
    ln_lambda_X = [X'*(digamma(bpmm.cmp[k].a) - log(bpmm.cmp[k].b)) for k in 1 : K]
    for n in 1 : N
        tmp_ln_pi =  [tmp[k] + ln_lambda_X[k][n] for k in 1 : K]
        ln_expt_Z[:,n] = tmp_ln_pi - logsumexp(tmp_ln_pi)
    end
    return ln_expt_Z
end

function winner_takes_all(Z::Matrix{Float64})
    Z_ret = zeros(size(Z))
    for n in 1 : size(Z_ret, 2)
        idx = indmax(Z[:,n])
        Z_ret[idx,n] = 1
    end
    return Z_ret
end

####################
## used for Gibbs Sampling
function sample_Z_GS(pmm::PMM, X::Matrix{Float64})
    D, N = size(X)
    K = pmm.K
    Z = zeros(K, N)

    tmp = [-sum(pmm.cmp[k].lambda) + log(pmm.phi[k]) for k in 1 : K]
    ln_lambda_X = [X'*log(pmm.cmp[k].lambda) for k in 1 : K]
    for n in 1 : N
        tmp_ln_phi = [(tmp[k] + ln_lambda_X[k][n])::Float64 for k in 1 : K]
        tmp_ln_phi = tmp_ln_phi - logsumexp(tmp_ln_phi)
        Z[:,n] = categorical_sample(exp(tmp_ln_phi))
    end
    return Z
end

####################
## used for Collapsed Gibbs Sampling
function calc_ln_NB(Xn::Vector{Float64}, gam::Gam)
    ln_lkh = [(gam.a[d]*log(gam.b)
               - lgamma(gam.a[d])
               + lgamma(Xn[d] + gam.a[d])
               - (Xn[d] + gam.a[d])*log(gam.b + 1)
               )::Float64 for d in 1 : size(Xn, 1)]
    return sum(ln_lkh)
end

function sample_Zn(Xn::Vector{Float64}, bpmm::BPMM)
    ln_tmp = [(calc_ln_NB(Xn, bpmm.cmp[k]) + log(bpmm.alpha[k])) for k in 1 : bpmm.K]
    ln_tmp = ln_tmp -  logsumexp(ln_tmp)
    Zn = categorical_sample(exp(ln_tmp))
    return Zn
end

function sample_Z_CGS(Z::Matrix{Float64}, X::Matrix{Float64}, bpmm::BPMM)
    D, N = size(X)
    K = size(Z, 1)
    for n in randperm(N)
        # remove
        bpmm = remove_stats(bpmm, X[:,[n]], Z[:,[n]])
        # sample
        Z[:,n] = sample_Zn(X[:,n], bpmm)
        # insert
        bpmm = add_stats(bpmm, X[:,[n]], Z[:,[n]])
    end
    return Z, bpmm
end

####################
## Algorithm main
function learn_BPMM_VI(X::Matrix{Float64}, prior_bpmm::BPMM, max_iter::Int)
    ## Variational Inference
    ## for Bayesian Poisson Mixture Model

    # initialisation
    expt_Z = init_Z(X, prior_bpmm)
    bpmm = add_stats(prior_bpmm, X, expt_Z)
    VB = NaN * zeros(max_iter)

    # inference
    for i in 1 : max_iter
        # E-step
        expt_Z = exp(update_Z(bpmm, X))
        # M-step
        bpmm = add_stats(prior_bpmm, X, expt_Z)
        # calc VB
        VB[i] = calc_bound(X, prior_bpmm, bpmm)
    end

    # assign binary values
    Z = winner_takes_all(expt_Z)
    return Z, bpmm, VB
end

function learn_BPMM_GS(X::Matrix{Float64}, prior_bpmm::BPMM, max_iter::Int)
    ## Gibbs Sampling
    ## for Bayesian Poisson Mixture Model

    # initialisation
    Z = init_Z(X, prior_bpmm)
    bpmm = add_stats(prior_bpmm, X, Z)
    VB = NaN * zeros(max_iter)
    
    # inference
    for i in 1 : max_iter            
        # sample parameters
        pmm = sample_PMM(bpmm)
        # sample latent variables
        Z = sample_Z_GS(pmm, X)
        # update current model
        bpmm = add_stats(prior_bpmm, X, Z)
        # calc VB
        VB[i] = calc_bound(X, prior_bpmm, bpmm)
    end

    return Z, bpmm, VB
end

function learn_BPMM_CGS(X::Matrix{Float64}, prior_bpmm::BPMM, max_iter::Int)
    ## Collapsed Gibbs Sampling
    ## for Bayesian Poisson Mixture Model
    
    # initialisation
    Z = init_Z(X, prior_bpmm)
    bpmm = add_stats(prior_bpmm, X, Z)
    VB = NaN * zeros(max_iter)

    # inference
    for i in 1 : max_iter
        # directly sample Z
        Z, bpmm = sample_Z_CGS(Z, X, bpmm)
        # calc VB
        VB[i] = calc_bound(X, prior_bpmm, bpmm)
    end

    return Z, bpmm, VB
end

