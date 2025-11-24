# Import necessary modules
#import Pkg; Pkg.add("JLD");Pkg.add("LinearAlgebra"), Pkg.add("DelimitedFiles"), Pkg.add("IterativeSolvers"), Pkg.add("Roots"), Pkg.add("SplitApplyCombine"),  Pkg.add("Distributions"), Pkg.add("LinearMaps"), Pkg.add("Random"),Pkg.add("CSV"),Pkg.add("DataFrames")
using JLD, LinearAlgebra, DelimitedFiles, Roots, Statistics, Distributions, IterativeSolvers, LinearMaps, Random, CSV, DataFrames
include("functions.jl")


function main()
    n = 1940
    Random.seed!(2)
    # Define the directory and file paths
    Z_train_file = "mouse_data.csv"
    # Read the CSV file into a data frame
    Z_train = CSV.read(Z_train_file, DataFrame; header=false)
    Zall = Matrix(Z_train[:, :]);  # Convert the DataFrame to a Matrix
    global m = 1015
    k = 100

    # Calculate the number of columns in each sub-matrix
    p = size(Zall, 2)
    cols_per_partition = div(p, m)
    remainder = mod(p, m)

    # Initialize the array to hold the sub-matrices
    Z = Vector{Matrix{Float64}}(undef, m)
    # Partition the matrix
    start_col = 1
    for i in 1:m
        end_col = start_col + cols_per_partition - 1
        if i <= remainder
            end_col += 1
        end
        Z[i] = Zall[:, start_col:end_col]
        start_col = end_col + 1
    end

    # Generate partition Array 
    grpsizes = [size(Z[i])[2] for i in 1:length(Z)]  # Array of partition sizes
    global grpsizes = Int.(grpsizes)

    #grpsizes = [n for i =1:m]
    p = sum(grpsizes)
    m = length(Z)
    # Initialize the start index
    start_index = 1
    # Create an empty dictionary to store the start and end indices for each partition
    indxs = Dict()
    # Calculate and store the indices for each partition
    for (i, grpsize) in enumerate(grpsizes)
        end_index = start_index + grpsize - 1
        indxs[i] = start_index: end_index
        start_index = end_index + 1  # Update the start index for the next partition
    end

    #ORTHOGONALIZE MATRICES
    Zq = Array{Matrix{Float64}}(undef, m)
    Zqall = zeros(n,p)
    Rs = Array{Matrix{Float64}}(undef, m)
    Z = center_columns_matrix_array(Z)
    Zall = center_columns(Zall)
        for i = 1:m
             #println(i)
             QRd = qr(Z[i])
             r = QRd.Q[:,1:grpsizes[i]]
             Zq[i] = r
             Rs[i] = QRd.R
        end
    Zqall = reduce(hcat, Zq);

    nsim = 25
    for isim in 1:nsim
        println("sim: $isim")
        betasall = zeros(1,p)
        betas = Array{Matrix{Float64}}(undef, m)
        indic =  Array{Integer}(undef, m)
        indic = spread_ones(m, k)
        for i = 1:m
             beta  = [-2,-1,0,1,2,-2,-1,0,1,2]
             #beta =[-2,-1,0,1,2]
             betas[i] = indic[i]*beta[:,:]
             betasall[indxs[i]] = betas[i]
        end
        Zt = Zall.-mean(Zall,dims=1);
        sigma = (Zt'*Zt)/(n-1);
        noise = betasall*sigma*betasall';


        y = zeros(n,1)    
        for i = 1:m
            y += Z[i]*betas[i]
        end

        y = y + sqrt(noise[1])*randn(n)
        y = vec(y)
        y = vec(y-mean(y)*ones(n,1));


        maxl = round(maximum([norm(Zq[j]'*y)/(sqrt(grpsizes[j])) for j = 1:m]),digits = 2)
        ngrid = 100
        minl = maxl*0.01
        lamrange = exp.(range(log(minl),log(maxl),length = ngrid))
        lamrange = reverse(lamrange)

        funtolall = 1e-5
        iters,actives,theta_mm,times_mm = group_lasso_mm(y,Zqall,verbose = false, maxiter = 20*10^3,funtol = funtolall,m=m,lambdas = lamrange,indxs = indxs, grpsizes = grpsizes);
        iters_cd,actives_cd,theta_cd,times_cd = group_lasso_bcd(y,Zq,Zqall,verbose = false, maxiter = 50000,funtol = funtolall,lambdas = lamrange, indxs = indxs, grpsizes = grpsizes);    
        writedlm("Mice_Setting1_Grp10"*"_sim_" * string(isim) * ".csv", [times_mm times_cd iters[:, 1] iters_cd[:, 1]], ",")

    end

end

main()
