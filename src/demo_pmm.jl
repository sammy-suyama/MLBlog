###################################
## Example code
## for Bayesian Poisson Mixture Model

using PyPlot, PyCall
include("PMM.jl")
#using Debug

function visualize_2D(X::Matrix{Float64}, Z::Matrix{Float64}, Z_est::Matrix{Float64})
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

function draw_hist(ax, X, Z, label)
    counts, bins, patches = ax[:hist](X', 20)
    for i in 1 : length(patches)
        if counts[i] > 0
            Z_tmp = Z[:,bins[i] .<= X[1,:] .<= bins[i+1]]
            Z_sum = sum(Z_tmp, 2) / sum(Z_tmp)
            patches[i][:set_facecolor]((Z_sum[1], 0, Z_sum[2]))
        end
    end
    ax[:set_title](label)
end

function visualize_1D(X::Matrix{Float64}, Z::Matrix{Float64}, Z_est::Matrix{Float64})
    # subplots
    #f, (ax1, ax2) = subplots(1,2)
    #draw_hist(ax1, X, Z, "truth")
    #draw_hist(ax2, X, Z_est, "estimation")

    # separated figures
    _, ax1 = subplots(1,1)
    _, ax2 = subplots(1,1)
    ax1[:hist](X', 20)
    ax1[:set_title]("observation")
    draw_hist(ax2, X, Z_est, "estimation")
    
end

function test_VB()
    ## set model
    D = 2 # data dimension
    K = 8 #  number of mixture components
    alpha = 1.0 * ones(K)
    cmp = [Gam(1.0*ones(D), 0.01) for i in 1 : K]
    bpmm = BPMM(D, K, alpha, cmp)
    
    ## generate data
    N = 1000
    pmm = sample_PMM(bpmm)
    X, Z = sample_data(pmm, N)
    
    ## inference
    #max_iter = 100
    max1=1250
    max2=192
    max3=160
    tic()
    _, _, VB1 = learn_BPMM_VI(X, bpmm, max1)
    toc()
    tic()
    _, _, VB2 = learn_BPMM_GS(X, bpmm, max2)
    toc()
    tic()
    _, _, VB3 = learn_BPMM_CGS(X, bpmm, max3)
    toc()

    # VB check
    figure()
    #plot(VB1);plot(VB2);plot(VB3);
    semilogx(linspace(1,10*1000,max1), VB1)
    semilogx(linspace(1,10*1000,max2), VB2)
    semilogx(linspace(1,10*1000,max3), VB3)
    legend(["VI","GS","CGS"], loc="left bottom")
    close()

    return VB1, VB2, VB3
end


function test_2D()
    ## set model
    D = 2 # data dimension
    K = 8 #  number of mixture components
    alpha = 100.0 * ones(K)
    cmp = [Gam(1.0*ones(D), 0.01) for i in 1 : K]
    bpmm = BPMM(D, K, alpha, cmp)
    
    ## generate data
    N = 100
    pmm = sample_PMM(bpmm)
    X, Z = sample_data(pmm, N)
    
    ## inference
    max_iter = 100
    tic()
    Z_est, post_bpmm, VB = learn_BPMM_VI(X, bpmm, max_iter)
    #Z_est, post_bpmm, VB = learn_BPMM_GS(X, bpmm, max_iter)
    #Z_est, post_bpmm, VB = learn_BPMM_CGS(X, bpmm, max_iter)
    toc()

    ## plot
    visualize_2D(X, Z, Z_est)

    # VB check
    println(VB)
    figure()
    plot(VB)
    show()
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
    X, Z = sample_data(pmm, N)
    
    ## inference
    max_iter = 20
    tic()
    Z_est, post_bpmm, VB = learn_BPMM_VI(X, bpmm, max_iter)
    #Z_est, post_bpmm, VB = learn_BPMM_GS(X, bpmm, max_iter)
    #Z_est, post_bpmm, VB = learn_BPMM_CGS(X, bpmm, max_iter)
    toc()

    ## plot
    visualize_1D(X, Z, Z_est)

    figure()
    plot(VB)
    show()
    
    println(VB)
    println()
end

#test_2D()
#test_1D()
#test_VB()


trials = 100
result1 = []
result2 = []
result3 = []
for _ in 1 : trials
    tmp = test_VB()
    push!(result1, tmp[1])
    push!(result2, tmp[2])
    push!(result3, tmp[3])
end

m_VB1 = mean(result1)
m_VB2 = mean(result2)
m_VB3 = mean(result3)

figure()
semilogx(linspace(1,10*1000,size(m_VB1, 1)), m_VB1)
semilogx(linspace(1,10*1000,size(m_VB2, 1)), m_VB2)
semilogx(linspace(1,10*1000,size(m_VB3, 1)), m_VB3)
legend(["VI","GS","CGS"], loc="lower right")
xlabel("time [msec]")
ylabel("lower bound")

