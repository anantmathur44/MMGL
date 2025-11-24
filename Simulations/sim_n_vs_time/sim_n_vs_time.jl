
# Import necessary modules
#import Pkg; Pkg.add("JLD");Pkg.add("LinearAlgebra"), Pkg.add("DelimitedFiles"), Pkg.add("IterativeSolvers"), Pkg.add("Roots"), Pkg.add("SplitApplyCombine"),  Pkg.add("Distributions"), Pkg.add("LinearMaps"), Pkg.add("Random")
using JLD, LinearAlgebra, DelimitedFiles, Roots, Statistics, Distributions, IterativeSolvers, LinearMaps, Random
#include("C:\\Users\\z5110079\\Desktop\\GitHub\\MMGL\\To_server\\functions.jl")
include("functions.jl")



function main()
    Random.seed!(2)

    grpsize = 3
    global m = 1000
    k = 100

    rhos = [0.3]
    psis = [0.3]
    SNR = 1
    nsim = 50
    ngrid = 100

    c  = size(rhos)[1]

    grpsizes = grpsize * ones(m)  # Array of partition sizes
    global grpsizes = Int.(grpsizes)
    # Initialize the start index
    start_index = 1
    # Create an empty dictionary to store the start and end indices for each partition
    indxs = Dict{Int, UnitRange{Int}}()
    # Calculate and store the indices for each partition
    for (i, grpsize) in enumerate(grpsizes)
        end_index = start_index + grpsize - 1
        indxs[i] = start_index:end_index
        start_index = end_index + 1  # Update the start index for the next partition
    end
    rho = rhos[1]
    psi = psis[1]
    sigma = gen_cov_matrix(m, grpsize, rho, psi)

    # SIMULATED DATA
    ns = range(2500,25000,10)
    for n in ns
        println("n: $n")
        for isim in 1:nsim
            println("sim: $isim")
            n = Int.(n)
            Z, Zall, betas, y, betasall, indic = simulate_data_gl(n, grpsize, m, SNR, sigma, k)
            y = vec(y - mean(y) * ones(n, 1))
            # ORTHOGONALIZE MATRICES
            Zq = Array{Matrix{Float64}}(undef, m)
            Zqall = zeros(n, grpsize * m)
            Rs = Array{Matrix{Float64}}(undef, m)
            Z = center_columns_matrix_array(Z)
            Zall = center_columns(Zall)
            for i = 1:m
                QRd = qr(Z[i])
                r = QRd.Q[:, 1:grpsize]
                Zq[i] = r
                Rs[i] = QRd.R
            end
            Zqall = reduce(hcat, Zq)
            maxl = round(maximum([norm(Zq[j]' * y)/(sqrt(grpsize)) for j in 1:m]), digits = 2)
            # minl = maxl*1e-04
            minl = maxl * 0.01
            lamrange = exp.(range(log(minl), log(maxl), length = ngrid))
            lamrange = reverse(lamrange)
            funtolall = 1e-5
            iters, actives, theta_mm, times_mm, gamma_mm = group_lasso_mm(y, Zqall; verbose = false, maxiter = 20 * 10^3, funtol = funtolall, m = m, lambdas = [lamrange[50]], indxs = indxs, grpsizes = grpsizes)
            #println("MM-Done")
            iters_bcd, actives_bcd, theta_bcd, times_bcd = group_lasso_bcd(y, Zq, Zqall; verbose = false, maxiter = 50000, funtol = funtolall, lambdas = [lamrange[50]], indxs = indxs, grpsizes = grpsizes)
            writedlm("n_time_" * string(n) *"_"*string(isim)* ".csv", [cumsum(iters) cumsum(iters_bcd) cumsum(times_mm) cumsum(times_bcd)], ",")
        end
    end
end

main()
