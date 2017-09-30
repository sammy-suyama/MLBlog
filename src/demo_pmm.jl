###################################
## Example code
## for Bayesian Poisson Mixture Model

push!(LOAD_PATH,".")
using PyPlot, PyCall
using PoissonMixtureModel

function visualize_2D(X::Matrix{Float64}, S::Matrix{Float64}, S_est::Matrix{Float64})
    cmp = get_cmap("jet")

    K1 = size(S, 1)
    K2 = size(S_est, 1)
    col1 = [pycall(cmp.o, PyAny, Int(round(val)))[1:3] for val in linspace(0,255,K1)]    
    col2 = [pycall(cmp.o, PyAny, Int(round(val)))[1:3] for val in linspace(0,255,K2)]    

    f, (ax1, ax2) = subplots(1,2)
    for k in 1 : K1
        ax1[:scatter](X[1, S[k,:].==1], X[2, S[k,:].==1], color=col1[k])
    end
    ax1[:set_title]("truth")
    
    for k in 1 : K2
        ax2[:scatter](X[1, S_est[k,:].==1], X[2, S_est[k,:].==1], color=col2[k])
    end

    ax2[:set_title]("estimation")
end

function draw_hist(ax, X, S, label)
    counts, bins, patches = ax[:hist](X', 20)
    for i in 1 : length(patches)
        if counts[i] > 0
            S_tmp = S[:,bins[i] .<= X[1,:] .<= bins[i+1]]
            S_sum = sum(S_tmp, 2) / sum(S_tmp)
            patches[i][:set_facecolor]((S_sum[1], 0, S_sum[2]))
        end
    end
    ax[:set_title](label)
end

function visualize_1D(X::Matrix{Float64}, S::Matrix{Float64}, S_est::Matrix{Float64})
    # separated figures
    _, ax1 = subplots(1,1)
    _, ax2 = subplots(1,1)
    ax1[:hist](X', 20)
    ax1[:set_title]("observation")
    draw_hist(ax2, X, S_est, "estimation")    
end

function test_2D()
    ## set model
    D = 2 # data dimension
    K = 4 #  number of mixture components
    alpha = 100.0 * ones(K)
    cmp = [Gam(1.0*ones(D), 0.01) for i in 1 : K]
    bpmm = BPMM(D, K, alpha, cmp)
    
    ## generate data
    N = 100
    pmm = sample_PMM(bpmm)
    X, S = sample_data(pmm, N)
    
    ## inference
    max_iter = 100
    tic()
    S_est, post_bpmm, VB = learn_BPMM_VI(X, bpmm, max_iter)
    #S_est, post_bpmm, VB = learn_BPMM_GS(X, bpmm, max_iter)
    #S_est, post_bpmm, VB = learn_BPMM_CGS(X, bpmm, max_iter)
    toc()

    ## plot
    visualize_2D(X, S, winner_takes_all(S_est))

    # VB check
    figure()
    plot(VB)
    ylabel("variational bound")
    xlabel("iterations")
end

function test_1D()
    ## set model
    D = 1 # data dimension
    K = 2 #  number of mixture components
    alpha = 100.0 * ones(K)
    cmp = [Gam(1.0*ones(D), 0.01) for i in 1 : K]
    bpmm = BPMM(D, K, alpha, cmp)
    
    ## generate data
    N = 1000
    pmm = sample_PMM(bpmm)
    X, S = sample_data(pmm, N)
    
    ## inference
    max_iter = 20
    tic()
    S_est, post_bpmm, VB = learn_BPMM_VI(X, bpmm, max_iter)
    #S_est, post_bpmm, VB = learn_BPMM_GS(X, bpmm, max_iter)
    #S_est, post_bpmm, VB = learn_BPMM_CGS(X, bpmm, max_iter)
    toc()

    ## plot
    visualize_1D(X, S, S_est)

    figure()
    plot(VB)
    ylabel("variational bound")
    xlabel("iterations")
end

#test_1D()
test_2D()

