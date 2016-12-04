###################################
## Bayesian Gaussian Mixture Model

import StatsFuns.logsumexp
include("sampler.jl")

####################
## Types
type GW
    # Parameters of Gauss Wisahrt distribution
    beta::Float64
    m::Vector{Float64}
    nu::Float64
    W::Matrix{Float64}
end

type BGMM
    # Parameters of Bayesian Gaussian Mixture Model 
    D::Int
    K::Int
    alpha::Vector{Float64}
    cmp::Vector{GW}
end

type Gauss
    # Parameters of Gauss Distribution
    mu::Vector{Float64}
    Lambda::Matrix{Float64}
end

type GMM
    # Parameters of Gauss Mixture Model
    D::Int
    K::Int
    phi::Vector{Float64}
    cmp::Vector{Gauss}
end

####################
## Common functions
function sample_GMM(bgmm::BGMM)
    cmp = Vector{Gauss}()
    for c in bgmm.cmp
        mu, Lambda = gw_sample(c.beta, c.m, c.nu, c.W)
        push!(cmp, Gauss(mu, Lambda))  
    end
    phi = dirichlet_sample(bgmm.alpha)
    return GMM(bgmm.D, bgmm.K, phi, cmp)
end

function sample_data(gmm::GMM, N::Int)
    X = zeros(gmm.D, N)
    Z = categorical_sample(gmm.phi, N)
    for n in 1 : N
        k = indmax(Z[:, n])
        X[:,n] = gaussian_sample(gmm.cmp[k].mu, inv(gmm.cmp[k].Lambda))
    end
    return X, Z
end

function init_Z(X::Matrix{Float64}, bgmm::BGMM)
    N = size(X, 2)
    K = bgmm.K
    Z = categorical_sample(ones(K)/K, N)    
    return Z
end

function calc_bound(X::Matrix{Float64}, pri::BGMM, pos::BGMM)
    ln_expt_Z = update_Z(pos, X)
    expt_Z = exp(ln_expt_Z)
    K, N = size(expt_Z)
    D = size(X, 1)

    expt_ln_lambda = zeros(D, K)
    expt_lambda = zeros(D, K)
    expt_lik = 0
    #for k in 1 : K
    #    expt_ln_lambda[:,k] = digamma(pos.cmp[k].a) - log(pos.cmp[k].b)
    #    expt_lambda[:,k] = pos.cmp[k].a / pos.cmp[k].b
    #    for n in 1 : N
    #        expt_lik += expt_Z[k,n] * (X[:, n]' * expt_ln_lambda[:,k]
    #                                   - sum(expt_lambda[:,k]) - sum(lgamma(X[:,n]+1)))[1]
    #    end
    #end
    
    #expt_ln_pZ = sum(expt_Z' * (digamma(pos.alpha) - digamma(sum(pos.alpha))))
    #expt_ln_qZ = sum(expt_Z .* ln_expt_Z)
    
    KL_lambda = 0
    #for k in 1 : K
    #    KL_lambda += (sum(pos.cmp[k].a)*log(pos.cmp[k].b) - sum(pri.cmp[k].a)*log(pri.cmp[k].b)
    #                  - sum(lgamma(pos.cmp[k].a)) + sum(lgamma(pri.cmp[k].a))
    #                  + (pos.cmp[k].a - pri.cmp[k].a)' * expt_ln_lambda[:,k]
    #                  + (pri.cmp[k].b - pos.cmp[k].b) * sum(expt_lambda[:,k])
    #                  )[1]
    end
    KL_pi = (lgamma(sum(pos.alpha)) - lgamma(sum(pri.alpha))
             - sum(lgamma(pos.alpha)) + sum(lgamma(pri.alpha))
             + (pos.alpha - pri.alpha)' * (digamma(pos.alpha) - digamma(sum(pos.alpha)))
             )[1]
    
    VB = expt_lik + expt_ln_pZ - expt_ln_qZ - (KL_lambda + KL_pi)
    return VB
end

function add_stats(bgmm::BGMM, X::Matrix{Float64}, Z::Matrix{Float64})
    D = bgmm.D
    K = bgmm.K
    sum_Z = sum(Z, 2)
    alpha = [bgmm.alpha[k] + sum_Z[k] for k in 1 : K]
    cmp = Vector{GW}()

    XZ = X*Z';
    for k in 1 : K
        #a = [(bgmm.cmp[k].a[d] + XZ[d,k])::Float64 for d in 1 : D]
        #b = bgmm.cmp[k].b + sum_Z[k]
        #push!(cmp, GW(a, b))
        push!(cmp, GW(beta, m, nu, W))
    end
    return BGMM(D, K, alpha, cmp)
end

remove_stats(bgmm::BGMM, X::Matrix{Float64}, Z::Matrix{Float64}) = add_stats(bgmm, X, -Z)

####################
## used for Variational Inference
function update_Z(bgmm::BGMM, X::Matrix{Float64})
    D, N = size(X)
    K = bgmm.K
    ln_expt_Z = zeros(K, N)
    tmp = zeros(K)

    sum_digamma_tmp = digamma(sum(bgmm.alpha))
    for k in 1 : K
        tmp[k] = - sum(bgmm.cmp[k].a) / bgmm.cmp[k].b
        tmp[k] += digamma(bgmm.alpha[k]) - sum_digamma_tmp
    end
    ln_lambda_X = [X'*(digamma(bgmm.cmp[k].a) - log(bgmm.cmp[k].b)) for k in 1 : K]
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
function sample_Z_GS(gmm::GMM, X::Matrix{Float64})
    D, N = size(X)
    K = gmm.K
    Z = zeros(K, N)

    tmp = [-sum(gmm.cmp[k].lambda) + log(gmm.phi[k]) for k in 1 : K]
    ln_lambda_X = [X'*log(gmm.cmp[k].lambda) for k in 1 : K]
    for n in 1 : N
        tmp_ln_phi = [(tmp[k] + ln_lambda_X[k][n])::Float64 for k in 1 : K]
        tmp_ln_phi = tmp_ln_phi - logsumexp(tmp_ln_phi)
        Z[:,n] = categorical_sample(exp(tmp_ln_phi))
    end
    return Z
end

####################
## used for Collapsed Gibbs Sampling
function calc_ln_NB(Xn::Vector{Float64}, gw::GW)
    #ln_lkh = [(gw.a[d]*log(gw.b)
    #           - lgamma(gw.a[d])
    #           + lgamma(Xn[d] + gw.a[d])
    #           - (Xn[d] + gw.a[d])*log(gw.b + 1)
    #           )::Float64 for d in 1 : size(Xn, 1)]
    return sum(ln_lkh)
end

function sample_Zn(Xn::Vector{Float64}, bgmm::BGMM)
    ln_tmp = [(calc_ln_NB(Xn, bgmm.cmp[k]) + log(bgmm.alpha[k])) for k in 1 : bgmm.K]
    ln_tmp = ln_tmp -  logsumexp(ln_tmp)
    Zn = categorical_sample(exp(ln_tmp))
    return Zn
end

function sample_Z_CGS(Z::Matrix{Float64}, X::Matrix{Float64}, bgmm::BGMM)
    D, N = size(X)
    K = size(Z, 1)
    for n in randperm(N)
        # remove
        bgmm = remove_stats(bgmm, X[:,[n]], Z[:,[n]])
        # sample
        Z[:,n] = sample_Zn(X[:,n], bgmm)
        # insert
        bgmm = add_stats(bgmm, X[:,[n]], Z[:,[n]])
    end
    return Z, bgmm
end

####################
## Algorithm main
function learn_BGMM_VI(X::Matrix{Float64}, prior_bgmm::BGMM, max_iter::Int)
    ## Variational Inference
    ## for Bayesian Gaussian Mixture Model

    # initialisation
    expt_Z = init_Z(X, prior_bgmm)
    bgmm = add_stats(prior_bgmm, X, expt_Z)
    VB = NaN * zeros(max_iter)

    # inference
    for i in 1 : max_iter
        # E-step
        expt_Z = exp(update_Z(bgmm, X))
        # M-step
        bgmm = add_stats(prior_bgmm, X, expt_Z)
        # calc VB
        VB[i] = calc_bound(X, prior_bgmm, bgmm)
    end

    # assign binary values
    Z = winner_takes_all(expt_Z)
    return Z, bgmm, VB
end

function learn_BGMM_GS(X::Matrix{Float64}, prior_bgmm::BGMM, max_iter::Int)
    ## Gibbs Sampling
    ## for Bayesian Gaussian Mixture Model

    # initialisation
    Z = init_Z(X, prior_bgmm)
    bgmm = add_stats(prior_bgmm, X, Z)
    VB = NaN * zeros(max_iter)
    
    # inference
    for i in 1 : max_iter            
        # sample parameters
        gmm = sample_GMM(bgmm)
        # sample latent variables
        Z = sample_Z_GS(gmm, X)
        # update current model
        bgmm = add_stats(prior_bgmm, X, Z)
        # calc VB
        VB[i] = calc_bound(X, prior_bgmm, bgmm)
    end

    return Z, bgmm, VB
end

function learn_BGMM_CGS(X::Matrix{Float64}, prior_bgmm::BGMM, max_iter::Int)
    ## Collapsed Gibbs Sampling
    ## for Bayesian Gaussian Mixture Model
    
    # initialisation
    Z = init_Z(X, prior_bgmm)
    bgmm = add_stats(prior_bgmm, X, Z)
    VB = NaN * zeros(max_iter)

    # inference
    for i in 1 : max_iter
        # directly sample Z
        Z, bgmm = sample_Z_CGS(Z, X, bgmm)
        # calc VB
        VB[i] = calc_bound(X, prior_bgmm, bgmm)
    end

    return Z, bgmm, VB
end

