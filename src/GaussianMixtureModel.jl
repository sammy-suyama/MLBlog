###################################
## Bayesian Gaussian Mixture Model

module GaussianMixtureModel

using StatsFuns.logsumexp
using Distributions
using PDMats

export GW, BGMM, Gauss, GMM
export sample_GMM, sample_data, winner_takes_all
export learn_BGMM_GS, learn_BGMM_CGS, learn_BGMM_VI

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
        Lambda = rand(Wishart(c.nu, PDMats.PDMat(Symmetric(c.W))))
        mu = rand(MvNormal(c.m, PDMats.PDMat(Symmetric(inv(c.beta*Lambda)))))
        push!(cmp, Gauss(mu, Lambda))
    end
    phi = rand(Dirichlet(bgmm.alpha))
    return GMM(bgmm.D, bgmm.K, phi, cmp)
end

function sample_data(gmm::GMM, N::Int)
    X = zeros(gmm.D, N)
    S = categorical_sample(gmm.phi, N)
    for n in 1 : N
        k = indmax(S[:, n])
        X[:,n] = rand(MvNormal(gmm.cmp[k].mu, PDMats.PDMat(Symmetric(inv(gmm.cmp[k].Lambda)))))
    end
    return X, S
end

categorical_sample(p::Vector{Float64}) = categorical_sample(p, 1)[:,1]
function categorical_sample(p::Vector{Float64}, N::Int)
    K = length(p)
    S = zeros(K, N)
    S_tmp = rand(Categorical(p), N)
    for k in 1 : K
        S[k,find(S_tmp.==k)] = 1
    end
    return S
end

function sumdigamma(nu, D)
    ret = 0.0
    for d in 1 : D
        ret += digamma(0.5*(nu + 1 - d))
    end
    return ret
end

function init_S(X::Matrix{Float64}, bgmm::BGMM)
    N = size(X, 2)
    K = bgmm.K
    S = categorical_sample(ones(K)/K, N)    
    return S
end

function calc_bound(X::Matrix{Float64}, pri::BGMM, pos::BGMM)
    ln_expt_S = update_S(pos, X)
    expt_S = exp(ln_expt_S)
    K, N = size(expt_S)
    D = size(X, 1)

    expt_ln_lambda = zeros(D, K)
    expt_lambda = zeros(D, K)
    expt_lik = 0
    #for k in 1 : K
    #    expt_ln_lambda[:,k] = digamma(pos.cmp[k].a) - log(pos.cmp[k].b)
    #    expt_lambda[:,k] = pos.cmp[k].a / pos.cmp[k].b
    #    for n in 1 : N
    #        expt_lik += expt_S[k,n] * (X[:, n]' * expt_ln_lambda[:,k]
    #                                   - sum(expt_lambda[:,k]) - sum(lgamma(X[:,n]+1)))[1]
    #    end
    #end
    
    #expt_ln_pS = sum(expt_S' * (digamma(pos.alpha) - digamma(sum(pos.alpha))))
    #expt_ln_qS = sum(expt_S .* ln_expt_S)
    
    KL_lambda = 0
    #for k in 1 : K
    #    KL_lambda += (sum(pos.cmp[k].a)*log(pos.cmp[k].b) - sum(pri.cmp[k].a)*log(pri.cmp[k].b)
    #                  - sum(lgamma(pos.cmp[k].a)) + sum(lgamma(pri.cmp[k].a))
    #                  + (pos.cmp[k].a - pri.cmp[k].a)' * expt_ln_lambda[:,k]
    #                  + (pri.cmp[k].b - pos.cmp[k].b) * sum(expt_lambda[:,k])
    #                  )[1]
    #end
    KL_pi = (lgamma(sum(pos.alpha)) - lgamma(sum(pri.alpha))
             - sum(lgamma(pos.alpha)) + sum(lgamma(pri.alpha))
             + (pos.alpha - pri.alpha)' * (digamma(pos.alpha) - digamma(sum(pos.alpha)))
             )[1]
    
    VB = expt_lik + expt_ln_pS - expt_ln_qS - (KL_lambda + KL_pi)
    return VB
end

function add_stats(bgmm::BGMM, X::Matrix{Float64}, S::Matrix{Float64})
    ## TODO:add stats????????
    
    D = bgmm.D
    K = bgmm.K
    sum_S = sum(S, 2)
    alpha = [bgmm.alpha[k] + sum_S[k] for k in 1 : K]
    cmp = Vector{GW}()

    XS = X*S';
    for k in 1 : K
        beta = bgmm.cmp[k].beta + sum_S[k]
        m = (1.0/beta)*(vec(X*S[[k],:]') + bgmm.cmp[k].beta*bgmm.cmp[k].m)
        nu = bgmm.cmp[k].nu + sum_S[k]
        W = inv(X*diagm(S[k,:])*X'
                       - beta*m*m'
                       + bgmm.cmp[k].beta*bgmm.cmp[k].m*bgmm.cmp[k].m'
                       + inv(bgmm.cmp[k].W))
        
        push!(cmp, GW(beta, m, nu, W))
    end
    return BGMM(D, K, alpha, cmp)
end

remove_stats(bgmm::BGMM, X::Matrix{Float64}, S::Matrix{Float64}) = add_stats(bgmm, X, -S)

####################
## used for Variational Inference
function update_S(bgmm::BGMM, X::Matrix{Float64})
    D, N = size(X)
    K = bgmm.K
    ln_S = zeros(K, N)
    tmp = zeros(K)

    tmp = NaN * zeros(K)
    sum_digamma_tmp = digamma(sum(bgmm.alpha))
    for k in 1 : K
        tmp[k] = -0.5*(bgmm.cmp[k].nu*trace(bgmm.cmp[k].m*bgmm.cmp[k].m'*bgmm.cmp[k].W)
                       + D*(1.0/bgmm.cmp[k].beta)
                       - (sumdigamma(bgmm.cmp[k].nu, D) + logdet(bgmm.cmp[k].W)))
        tmp[k] += digamma(bgmm.alpha[k]) - sum_digamma_tmp
    end
    for n in 1 : N
        tmp_ln_pi = NaN * zeros(K)
        for k in 1 : K
            tmp_ln_pi[k] = tmp[k] -0.5*bgmm.cmp[k].nu*trace((X[:,n]*X[:,n]' - 2*bgmm.cmp[k].m*X[:,n]')*bgmm.cmp[k].W)
        end
        ln_S[:,n] = tmp_ln_pi - logsumexp(tmp_ln_pi)
    end
    return ln_S    
end

function winner_takes_all(S::Matrix{Float64})
    S_ret = zeros(size(S))
    for n in 1 : size(S_ret, 2)
        idx = indmax(S[:,n])
        S_ret[idx,n] = 1
    end
    return S_ret
end

####################
## used for Gibbs Sampling
function sample_S_GS(gmm::GMM, X::Matrix{Float64})
    D, N = size(X)
    K = gmm.K
    S = zeros(K, N)

    tmp = [-sum(gmm.cmp[k].lambda) + log(gmm.phi[k]) for k in 1 : K]
    ln_lambda_X = [X'*log(gmm.cmp[k].lambda) for k in 1 : K]
    for n in 1 : N
        tmp_ln_phi = [(tmp[k] + ln_lambda_X[k][n])::Float64 for k in 1 : K]
        tmp_ln_phi = tmp_ln_phi - logsumexp(tmp_ln_phi)
        S[:,n] = categorical_sample(exp(tmp_ln_phi))
    end
    return S
end

####################
## used for Collapsed Gibbs Sampling
function calc_ln_ST(Xn::Vector{Float64}, gw::GW)
    # TODO; need to check value?
    D = size(Xn, 1)
    W = ((1 - D + gw.nu)*gw.beta / (1 + gw.beta)) * gw.W
    ln_lkh = logpdf(MvTDist(1 - D + gw.nu, gw.m, (gw.nu/(gw.nu - 2))*inv(W)), Xn)
    return sum(ln_lkh)
end

function sample_Sn(Xn::Vector{Float64}, bgmm::BGMM)
    ln_tmp = [(calc_ln_ST(Xn, bgmm.cmp[k]) + log(bgmm.alpha[k])) for k in 1 : bgmm.K]
    ln_tmp = ln_tmp -  logsumexp(ln_tmp)
    Sn = categorical_sample(exp(ln_tmp))
    return Sn
end

function sample_S_CGS(S::Matrix{Float64}, X::Matrix{Float64}, bgmm::BGMM)
    D, N = size(X)
    K = size(S, 1)
    for n in randperm(N)
        # remove
        bgmm = remove_stats(bgmm, X[:,[n]], S[:,[n]])
        # sample
        S[:,n] = sample_Sn(X[:,n], bgmm)
        # insert
        bgmm = add_stats(bgmm, X[:,[n]], S[:,[n]])
    end
    return S, bgmm
end

####################
## Algorithm main
function learn_BGMM_VI(X::Matrix{Float64}, prior_bgmm::BGMM, max_iter::Int)
    ## Variational Inference
    ## for Bayesian Gaussian Mixture Model

    # initialisation
    expt_S = init_S(X, prior_bgmm)
    bgmm = add_stats(prior_bgmm, X, expt_S)
    VB = NaN * zeros(max_iter)

    # inference
    for i in 1 : max_iter
        # E-step
        expt_S = exp(update_S(bgmm, X))
        # M-step
        bgmm = add_stats(prior_bgmm, X, expt_S)
        # calc VB
        # TODO
        #VB[i] = calc_bound(X, prior_bgmm, bgmm)
    end

    # assign binary values
    S = winner_takes_all(expt_S)
    return S, bgmm, VB
end

function learn_BGMM_GS(X::Matrix{Float64}, prior_bgmm::BGMM, max_iter::Int)
    ## Gibbs Sampling
    ## for Bayesian Gaussian Mixture Model

    # initialisation
    S = init_S(X, prior_bgmm)
    bgmm = add_stats(prior_bgmm, X, S)
    VB = NaN * zeros(max_iter)
    
    # inference
    for i in 1 : max_iter            
        # sample parameters
        gmm = sample_GMM(bgmm)
        # sample latent variables
        S = sample_S_GS(gmm, X)
        # update current model
        bgmm = add_stats(prior_bgmm, X, S)
        # calc VB
        # TODO:
        #VB[i] = calc_bound(X, prior_bgmm, bgmm)
    end

    return S, bgmm, VB
end

function learn_BGMM_CGS(X::Matrix{Float64}, prior_bgmm::BGMM, max_iter::Int)
    ## Collapsed Gibbs Sampling
    ## for Bayesian Gaussian Mixture Model
    ## TODO
    
    # initialisation
    S = init_S(X, prior_bgmm)
    bgmm = add_stats(prior_bgmm, X, S)
    VB = NaN * zeros(max_iter)

    # inference
    for i in 1 : max_iter
        # directly sample S
        S, bgmm = sample_S_CGS(S, X, bgmm)
        # calc VB
        # TODO
        #VB[i] = calc_bound(X, prior_bgmm, bgmm)
    end

    return S, bgmm, VB
end

end
