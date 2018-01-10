###################################
## Bayesian Deep Learning
## via Gaussian Process Regression

using Distributions
using PDMats
using PyPlot

function calc_kernel_matrix(X, kernel)
    N = length(X)
    K = zeros(N, N)
    for n1 in 1 : N
        for n2 in n1 : N
            K[n1, n2] = kernel(X[n1], X[n2])
        end
    end
    return Symmetric(K)
end

function calc_kernel_sequence(X, x, kernel)
    N = length(X)
    seq = [kernel(X[n], x) for n in 1 : N]
    return seq
end

function predict(X_train, Y_train, X, kernel, sigma2_y)
    N = length(X)
    mu = zeros(N)
    sigma2 = zeros(N)
    N_train = length(Y_train)
    K = calc_kernel_matrix(X_train, kernel)
    invmat = inv(sigma2_y * eye(N_train) + K)
    for n in 1 : N
        seq = calc_kernel_sequence(X_train, X[n], kernel) # N dim
        mu[n] = seq' * invmat * Y_train
        sigma2[n] = sigma2_y + kernel(X[n], X[n]) - seq'*invmat*seq
    end
    
    return mu, sigma2
end

function plot_predict(X, mu, sigma2)
    fill_between(X, mu + sqrt.(sigma2), mu - sqrt.(sigma2), color="c", alpha=0.5)
    plot(X, mu, "-b")
end


###################
# common setting

# input
x_min = - 5.0
x_max = + 5.0
N = 100
X = linspace(x_min, x_max, N)

# noise parameter
sigma2_y = 0.1

###################
# kernels

# multinomial covariance function
M = 3
Sigma_W = diagm([10.0, 1.0, 0.1, 0.01])
multi(x, M) = [x^m for m in 0:M]
kernel_m(x1, x2) = trace(multi(x1, M)*multi(x2, M)'*Sigma_W)
K_m = calc_kernel_matrix(X, kernel_m)

# RBF covariance function
alpha = 1.0
beta = 1.0
kernel_g(x1, x2) = alpha * exp(-0.5*inv(beta^2)*(x1 - x2)^2)
K_g = calc_kernel_matrix(X, kernel_g)

# neural network (erf) covariance function
Sigma = 10.0*eye(2)
aug(x) = Array([1, x])
kernel_e(x1, x2) = (2/pi)*asin((2*trace(aug(x2)*aug(x1)'*Sigma))/sqrt((1+2*trace(aug(x1)*aug(x1)'*Sigma))*(1+2*trace(aug(x2)*aug(x2)'*Sigma))))
K_e = calc_kernel_matrix(X, kernel_e)

# neural network (ReLU) covariance function
rad(x1, x2) = acos(max(min(sum(x1.*x2)/(norm(x1)*norm(x2)), 1.0), -1.0))
kernel_r(x1, x2) = (1.0 / pi) * norm(aug(x1)) * norm(aug(x2)) * (sin(rad(aug(x1), aug(x2))) + (pi - rad(aug(x1), aug(x2)))*cos(rad(aug(x1), aug(x2))))
K_r = calc_kernel_matrix(X, kernel_r)

# deep neural network (ReLU) covariance function
L = 8
sigma2_b = 1.0
sigma2_w = 2.0
kernel_tmp(x1, x2, L) = L > 0 ? (k_11=kernel_tmp(x1, x1, L-1);k_22=kernel_tmp(x2, x2, L-1);theta=acos(kernel_tmp(x1, x2, L-1) / sqrt(k_11 * k_22));sigma2_b + (sigma2_w/(2*pi))*sqrt(k_11*k_22)*(sin(theta) + (pi - theta)*cos(theta)) ) : sigma2_b + sigma2_w*(x1'*x2 / length(x1))
kernel_d(x1, x2) = kernel_tmp(x1, x2, L)
K_d = calc_kernel_matrix(X, kernel_d)

###################
# training

# data
N_train_all = 40
X_train_all = linspace(-3.0, 3.0, N_train_all)
Y_train_all = sin.(X_train_all .* 2*pi / (maximum(X_train_all) - minimum(X_train_all)))

# sample from priors
Y_m = []
Y_g = []
Y_e = []
Y_r = []
Y_d = []
tiny = 1.0e-5
num_sample = 10
for _ in 1 : num_sample
    push!(Y_m, rand(MvNormal(zeros(N), tiny*eye(N) + K_m)))
    push!(Y_g, rand(MvNormal(zeros(N), tiny*eye(N) + K_g)))
    push!(Y_e, rand(MvNormal(zeros(N), tiny*eye(N) + K_e)))
    push!(Y_r, rand(MvNormal(zeros(N), tiny*eye(N) + K_r)))
    push!(Y_d, rand(MvNormal(zeros(N), tiny*eye(N) + K_d)))
end

dir_name = "fig"
if !isdir(dir_name)
    mkdir(dir_name)
end

figure("function sample", figsize=(16,8))
for n in 1 : N_train_all
    clf()

    N_train = n
    X_train = X_train_all[1:n]
    Y_train = Y_train_all[1:n]

    # prediction
    mu_m, sigma2_m = predict(X_train, Y_train, X, kernel_m, sigma2_y)
    mu_g, sigma2_g = predict(X_train, Y_train, X, kernel_g, sigma2_y)
    mu_e, sigma2_e = predict(X_train, Y_train, X, kernel_e, sigma2_y)
    mu_r, sigma2_r = predict(X_train, Y_train, X, kernel_r, sigma2_y)
    mu_d, sigma2_d = predict(X_train, Y_train, X, kernel_d, sigma2_y)
    
    ###################
    # visualization

    # prior
    for i in 1 : num_sample
        # sample Y
        subplot(2,5,1)
        title("cubic")
        plot(X, Y_m[i], "-")
        
        subplot(2,5,2)
        title("RBF")
        plot(X, Y_g[i], "-")
        
        subplot(2,5,3)
        title("NN (erf)")
        plot(X, Y_e[i], "-")
        
        subplot(2,5,4)
        title("NN (ReLU)")
        plot(X, Y_r[i], "-")

        subplot(2,5,5)
        title("DNN (ReLU)")
        plot(X, Y_d[i], "-")
    end
    
    # prediction
    y_min = -3
    y_max = +3
    
    subplot(2,5,6)
    plot_predict(X, mu_m, sigma2_m)
    plot(X_train, Y_train, "xk")
    xlim([x_min, x_max])
    ylim([y_min, y_max])
    
    subplot(2,5,7)
    plot_predict(X, mu_g, sigma2_g)
    plot(X_train, Y_train, "xk")
    xlim([x_min, x_max])
    ylim([y_min, y_max])
    
    subplot(2,5,8)
    plot_predict(X, mu_e, sigma2_e)
    plot(X_train, Y_train, "xk")
    xlim([x_min, x_max])
    ylim([y_min, y_max])
    
    subplot(2,5,9)
    plot_predict(X, mu_r, sigma2_r)
    plot(X_train, Y_train, "xk")
    xlim([x_min, x_max])
    ylim([y_min, y_max])

    subplot(2,5,10)
    plot_predict(X, mu_d, sigma2_d)
    plot(X_train, Y_train, "xk")
    xlim([x_min, x_max])
    ylim([y_min, y_max])

    savefig(@sprintf("fig/%03d.png", n))
end

