#
# Metropolis-Hastings for logistic regression
# with missing input values
#

using PyPlot, PyCall
using Distributions
PyDict(matplotlib["rcParams"])["mathtext.fontset"] = "cm"
PyDict(matplotlib["rcParams"])["mathtext.rm"] = "serif"
PyDict(matplotlib["rcParams"])["lines.linewidth"] = 1.5
PyDict(matplotlib["rcParams"])["font.family"] = "TakaoPGothic"

function visualize_contour(samples_W, X, Y, text)
    R = 100
    xmin = minimum(X[1,:])
    xmax = maximum(X[1,:])
    ymin = minimum(X[2,:])
    ymax = maximum(X[2,:])
    lx = xmax - xmin
    ly = ymax - ymin
    xmin = xmin - 0.25 * lx
    xmax = xmax + 0.25 * lx
    ymin = ymin - 0.25 * ly
    ymax = ymax + 0.25 * ly

    x1 = linspace(xmin,xmax,R)
    x2 = linspace(ymin,ymax,R)
    x1grid = repmat(x1, 1, R)
    x2grid = repmat(x2', R, 1)
    val = [x1grid[:] x2grid[:]]'

    z_list = []
    for n in 1 : size(samples_W, 2)
        W = samples_W[:,n]
        z_tmp = [sigmoid(W'*val[:,i]) for i in 1 : size(val, 2)]
        push!(z_list, z_tmp)
    end
    z = mean(z_list)

    zgrid = reshape(z, R, R)

    # precition
    contour(x1grid, x2grid, zgrid, alpha=0.5, cmap=get_cmap("bwr"))
    scatter(X[1,Y.==1], X[2,Y.==1], c="r")
    scatter(X[1,Y.==0], X[2,Y.==0], c="b")
    xlim([xmin, xmax])
    ylim([ymin, ymax])    
    title("$text")
end

function draw_line(W, xmin, xmax)
    y1 = - xmin*W[1]/W[2]
    y2 = - xmax*W[1]/W[2]
    plot([xmin, xmax], [y1, y2], c="k")
end

function sigmoid(x)
    return 1.0 / (1.0 + exp(-x[1]))
end

function bern_sample(mu)
    i = rand(Bernoulli(mu))
    val = zeros(2)
    val[i+1] = 1
    return val
end

function insert_X(X, X_miss)
    X_copy = deepcopy(X)
    X_copy[isnan(X)] = X_miss
    return X_copy
end

function MH(Y, X, M, Sigma_w, sigma_x, sigma_mh, max_iter, burnin)
    function ln_p_tilde(W, Y, X, Sigma_w, sigma_x)
        lkh = [Y[n]*log(sigmoid(W'*X[:, n])) + (1 - Y[n])*log(1 - sigmoid(W'*X[:, n])) for n in 1 : size(X, 2)]
        pri_w = -0.5 * (W'*inv(Sigma_w)*W)[1]
        pri_x = -0.5 * (sum(X.^2) * inv(sigma_x^2)) # Warning: Observation is not needed.
        return sum(lkh) + pri_w + pri_x
    end

    M = size(X, 1)
    N_miss = sum(isnan(X))
    
    ## sampling
    W = randn(M)
    X_miss = randn(N_miss)
    
    samples_W = Array{Float64, 2}(M, max_iter)
    samples_X = Array{Float64, 2}(N_miss, max_iter)
    for i in 1 : max_iter
        # sample W
        W_new = rand(MvNormal(W, sigma_mh^2*eye(M)))
        X_tmp = insert_X(X, X_miss)
        r = exp(minimum([ln_p_tilde(W_new, Y, X_tmp, Sigma_w, sigma_x) - ln_p_tilde(W, Y, X_tmp, Sigma_w, sigma_x), 0]))
        W = (rand() < r) ? W_new : W
        samples_W[:, i] = W

        # sample X_miss
        X_miss_new = rand(MvNormal(X_miss, sigma_mh^2*eye(N_miss)))
        X_tmp1 = insert_X(X, X_miss_new)
        X_tmp2 =  insert_X(X, X_miss)
        r = exp(minimum([ln_p_tilde(W, Y, X_tmp1, Sigma_w, sigma_x) - ln_p_tilde(W, Y, X_tmp2, Sigma_w, sigma_x), 0]))
        X_miss = (rand() < r) ? X_miss_new : X_miss
        samples_X[:, i] = X_miss
    end
    
    return samples_W[:,burnin+1:end], samples_X[:,burnin+1:end]
end

function test()
    ## compare results by discarding, interpolation and full observatino

    ########################
    # create model
    
    # hyperparameters
    M = 2
    Sigma_w = 20.0 * eye(M)
    
    ########################
    # create toy-data
    
    # sample parameters
    W = rand(MvNormal(zeros(M), Sigma_w))
    sigma_x = 1.0
    
    # sample data
    N = 30
    X_raw = 2 * rand(M, N) - 1.0
    Y = [rand(Bernoulli(sigmoid(W'*X_raw[:, n]))) for n in 1 : N]
    
    # missing values
    rate = 0.50
    X = deepcopy(X_raw)
    for n in 1 : N
        for m in 1 : M
            X[m,n] = (rand() < rate) ? NaN : X[m,n]
        end
    end

    idx = vec(!any(isnan(X), 1))
    X_red = X[:,idx]
    Y_red = Y[idx]
    
    ########################
    # inference
    sigma_mh = 0.10
    max_iter = 4000
    burnin = 1000

    samples_W1, samples_X1 = MH(Y_red, X_red, M, Sigma_w, sigma_x, sigma_mh, max_iter, burnin)
    samples_W2, samples_X2 = MH(Y, X, M, Sigma_w, sigma_x, sigma_mh, max_iter, burnin)
    samples_W3, samples_X3 = MH(Y, X_raw, M, Sigma_w, sigma_x, sigma_mh, max_iter, burnin)
    
    ########################
    # summarize
    idx_nan = vec(any(isnan(X), 1))
    X_est = insert_X(X, mean(samples_X2, 2))
    X_std = 0*deepcopy(X)
    X_std = insert_X(X_std, std(samples_X2, 2))

    ########################
    # visualize
    figure("result", figsize=(12,3))
    clf()
    subplot(1,3,1)
    visualize_contour(samples_W1, X_red, Y_red, "Reduced data (N=$(length(Y_red)))")
    
    subplot(1,3,2)
    Y_tmp = Y[idx_nan]
    X_tmp = X_est[:, idx_nan]
    scatter(X_tmp[1,Y_tmp.==1], X_tmp[2,Y_tmp.==1], c="r", marker="x")
    scatter(X_tmp[1,Y_tmp.==0], X_tmp[2,Y_tmp.==0], c="b", marker="x")
    for i in find(idx_nan)
        if Y[i] == 1
            plot([X_est[1,i] + X_std[1,i], X_est[1,i] - X_std[1,i]], [X_est[2,i], X_est[2,i]], "r", alpha=0.20, linewidth=5)
            plot([X_est[1,i], X_est[1,i]], [X_est[2,i] + X_std[2,i], X_est[2,i] - X_std[2,i]], "r", alpha=0.20, linewidth=5)
        else
            plot([X_est[1,i] + X_std[1,i], X_est[1,i] - X_std[1,i]], [X_est[2,i], X_est[2,i]], "b", alpha=0.20, linewidth=5)
            plot([X_est[1,i], X_est[1,i]], [X_est[2,i] + X_std[2,i], X_est[2,i] - X_std[2,i]], "b", alpha=0.20, linewidth=5)
        end
    end
    visualize_contour(samples_W2, X, Y, "Incomplete data (N=$N)")

    subplot(1,3,3)
    visualize_contour(samples_W3, X_raw, Y, "Full data (N=$N)")

    figure("traces", figsize=(12, 12))
    clf()
    subplot(4,1,1)
    plot(samples_W1')
    subplot(4,1,2)
    plot(samples_W2')
    subplot(4,1,3)
    plot(samples_X2')
    subplot(4,1,4)
    plot(samples_W3')
    
end

test()
