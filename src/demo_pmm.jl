###################################
## Example code
## for Bayesian Poisson Mixture Model

using PyPlot, PyCall
include("PMM.jl")
using Debug

@debug function visualize(X::Matrix{Float64}, Z::Matrix{Float64}, Z_est::Matrix{Float64})
    #col = ["r","g","b","c","m", "y"]
    cmp = get_cmap("jet")

    K1 = size(Z, 1)
    K2 = size(Z_est, 1)
    col1 = [pycall(cmp.o, PyAny, Int(round(val)))[1:3] for val in linspace(0,255,K1)]    
    col2 = [pycall(cmp.o, PyAny, Int(round(val)))[1:3] for val in linspace(0,255,K2)]    

    f, (ax1, ax2) = subplots(1,2)
    for k in 1 : K1
        ax1[:scatter](X[1, Z[k,:].==1], X[2, Z[k,:].==1], color=col1[k])
    end
    ax1[:set_title]("truth")
    
    for k in 1 : K2
        ax2[:scatter](X[1, Z_est[k,:].==1], X[2, Z_est[k,:].==1], color=col2[k])
    end
    ax2[:set_title]("estimation")
end

function test()
    ## set model
    D = 2 # data dimension
    K = 8 #  number of mixture components
    alpha = 100.0 * ones(K)
    cmp = [Gam(1.0*ones(D), 0.01) for i in 1 : K]
    bpmm = BPMM(D, K, alpha, cmp)
    
    ## generate data
    N = 1000
    pmm = sample_PMM(bpmm)
    X, Z = sample_data(pmm, N)
    
    ## inference
    max_iter = 100
    tic()
    #Z_est, post_bpmm = learn_BPMM_VI(X, bpmm, max_iter)
    #Z_est, post_bpmm = learn_BPMM_GS(X, bpmm, max_iter)
    Z_est, post_bpmm = learn_BPMM_CGS(X, bpmm, max_iter)
    toc()

    ## plot
    visualize(X, Z, Z_est)
    println()
end

test()
