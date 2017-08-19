using PyPlot, PyCall
using Distributions
using StatsFuns
using Requests
PyDict(matplotlib["rcParams"])["mathtext.fontset"] = "cm"
PyDict(matplotlib["rcParams"])["mathtext.rm"] = "serif"
PyDict(matplotlib["rcParams"])["lines.linewidth"] = 1.5
PyDict(matplotlib["rcParams"])["font.family"] = "TakaoPGothic"

test = false

function VI(C, a1, b1, a2, b2, phi, max_iter)
    N = size(C, 1)
    phi_h = deepcopy(phi)

    a1_h = NaN
    b1_h = NaN
    a2_h = NaN
    b2_h = NaN
    for i in 1 : max_iter
        # update parameters
        phi_sum = sum(phi_h)

        #delta1 = reverse(cumsum(reverse(phi_h)))
        delta1 = reverse(cumsum(reverse(cat(1, phi_h[2:end], 0))))
        #tmp = cumsum(phi_h)
        #delta2 = cat(1, 0, tmp[1:end-1])
        delta2 = cumsum(phi_h)
        a1_h = a1 + sum(C.*delta1)
        b1_h = b1 + sum(delta1)
        a2_h = a2 + sum(C.*delta2)
        b2_h = b2 + sum(delta2)

        # update tau
        ln_p1 = Array{Float64, 1}(N)
        ln_p2 = Array{Float64, 1}(N)
        for n in 1 : N
            ln_p1[n] = C[n] * (digamma(a1_h) - log(b1_h)) - a1_h/b1_h
            ln_p2[n] = C[n] * (digamma(a2_h) - log(b2_h)) - a2_h/b2_h
        end

        ln_tmp = Array{Float64, 1}(N)
        for n in 1 : N
            #ln_tmp[n] = sum(ln_p1[1:n]) + sum(ln_p2[n+1:end]) + log(phi[n])
            ln_tmp[n] = sum(ln_p1[1:n-1]) + sum(ln_p2[n:end]) + log(phi[n])
        end
        phi_h = exp(ln_tmp - logsumexp(ln_tmp))
    end
    return phi_h, a1_h / b1_h, a2_h / b2_h
end

# model params
a1 = 2.0
b1 = 0.1
a2 = 2.0
b2 = 0.1

# generate or download data
if test
    N = 74
    phi = ones(N) / N

    # generate sample
    lambda1 = rand(Gamma(a1, 1.0/b1))
    lambda2 = rand(Gamma(a2, 1.0/b2))
    tau = rand(Categorical(phi))
    C = Array{Float64, 1}(N)
    for n in 1 : tau
        C[n] = rand(Poisson(lambda1))
    end
    for n in tau+1 : N
        C[n] = rand(Poisson(lambda2))
    end
else
    data_url = "https://git.io/vXTVC"
    dir_name = "data"
    file_name = "txtdata.csv"
    if !isdir(dir_name)
        mkdir(dir_name)
    end
    if isfile(dir_name * "/" * file_name)
        println("read csv file $(dir_name * "/" * file_name)")
        data = readcsv(dir_name * "/" * file_name)
    else
        println("download csv file from $data_url")
        res = get(data_url)
        data = readcsv(IOBuffer(res.data))
        writecsv(dir_name * "/" * file_name, data)
    end
    C = data[:,1]

    N = size(C, 1)
    phi = ones(N) / N
end



# visualize
if test
    figure("data samples")
    clf()
    bar(1:N, C, color="#348ABD")
    plot([1, tau], [lambda1, lambda1], "g--")
    plot([tau+1, N], [lambda2, lambda2], "g--")
    plot([tau+0.5, tau+0.5],[0, maximum(C)],"r--")
    xlim([0, N])
else
    figure("data samples")
    clf()
    bar(1:N, C, color="#348ABD")
    xlim([0, N])
end

# inference
max_iter = 100
phi_est, lambda1_est, lambda2_est = VI(C, a1, b1, a2, b2, phi, max_iter)

# calc expectation
tau_expt = sum(phi_est .* range(1,N))
lambda_est = Array{Float64, 1}(N)
for n in 1 : N
    tmp = sum(phi_est[1:n])
    lambda_est[n] =(1.0 - tmp) * lambda1_est + tmp * lambda2_est
end

figure("results")
clf()
subplot(2,1,1)
bar(1:N, C, color="#348ABD", label="受信数")
plot(1:N, lambda_est, color="#E24A33", label="期待値")
xlim([0, N])
ylabel("受信メッセージ数")
xlabel("経過日数")
legend()
subplot(2,1,2)
bar(1:N, phi_est, color="#348ABD", label="事後確率")
plot([tau_expt, tau_expt], [0, maximum(phi_est)], "r--", label="期待値")
xlim([0, N])
ylabel("事後確率")
xlabel("経過日数")
legend()

