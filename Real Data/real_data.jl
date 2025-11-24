using JLD,LinearAlgebra, DelimitedFiles, Roots, Statistics, Distributions, IterativeSolvers, LinearMaps, Random, CSV, DataFrames

include("functions.jl")


function compute_indices(grpsizes)
    start_index = 1             
    indxs = Dict{Int, UnitRange{Int}}()  # specify type if possible for clarity
    for (i, grpsize) in enumerate(grpsizes)
        end_index = start_index + grpsize - 1
        indxs[i] = start_index:end_index
        start_index = end_index + 1  # update local variable
    end
    return indxs
end


# Read the CSV files into data frames without headers
Z_train = CSV.read("x_train.csv", DataFrame)
grp = CSV.read("group_ids.csv", DataFrame; header=false)
y_train = CSV.read("y_train.csv", DataFrame; header=false)

# Convert DataFrames to Matrices
Z_train = Matrix(Z_train)
grp = Matrix(grp)
y_train = Matrix(y_train)

# Combine grp vector with column indices and sort by grp
grp_with_indices = collect(enumerate(grp))
sort!(grp_with_indices, by = x -> x[2],dims =1)

# Extract sorted indices and use them to reorder columns of Z_train_matrix
sorted_indices = [x[1] for x in grp_with_indices]
Z_train_sorted = Z_train[:, sorted_indices]

# Sorted grp vector
grp_sorted = grp[sorted_indices]

# Find unique group identifiers
unique_groups = unique(grp_sorted)

# Initialize an empty array to store the grouped matrices
Z = []

# Loop through each unique group and extract corresponding columns
for group in unique_groups
    # Find the columns that belong to the current group
    group_indices = findall(x -> x == group, grp_sorted)
    
    # Extract the columns and create a matrix
    group_matrix = Z_train_sorted[:, group_indices]
    
    # Append the matrix to the list
    push!(Z, group_matrix)
end

Zall = Z_train_sorted[:,:]

n = 418
# Generate partition Array 
grpsizes = [size(Z[i])[2] for i in 1:length(Z)]  # Array of partition sizes
grpsizes = Int.(grpsizes)

#grpsizes = [n for i =1:m]
p = sum(grpsizes)
m = length(Z)
# Initialize the start index

indxs = compute_indices(grpsizes)
y = y_train
y = vec(y-mean(y)*ones(n,1));


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

#Create grid
maxl = round(maximum([norm(Zq[j]'*y)/(sqrt(grpsizes[j])) for j = 1:m]),digits = 2)
ngrid = 100
minl = maxl*0.01
lamrange = exp.(range(log(minl),log(maxl),length = ngrid))
lamrange = reverse(lamrange)

funtolall = 1e-5
#MM
iters,actives,beta_mm,times_mm = group_lasso_mm(y,Zqall,verbose = false, maxiter = 20*10^3,funtol = funtolall,m=m,lambdas = lamrange,indxs = indxs, grpsizes = grpsizes);

#BCD Method 
iters_bcd,actives_bcd,beta_bcd,times_bcd = group_lasso_bcd(y,Zq,Zqall,verbose = false, maxiter = 50000,funtol = funtolall,lambdas = lamrange, indxs = indxs, grpsizes = grpsizes);

# Define the directory and file paths

# Read the CSV files into data frames without headers
Z_test = CSV.read("xs_test.csv", DataFrame; header=false)
y_test = CSV.read("y_test.csv", DataFrame; header=false)

# Convert DataFrames to Matrices
Z_test = Matrix(Z_test);
y_test = Matrix(y_test);
ntest = size(y_test)[1];

y_test = vec(y_test-mean(y_test)*ones(ntest,1));

#Obtain Testing Losses
lt_bcd = []
for k in 1:ngrid
    lt = loss_test(Z_test, y_test, beta_bcd[:,k], grpsizes, indxs, ntest,Rs,p)   
    append!( lt_bcd, lt )
end

lt_mm = []
for k in 1:ngrid
    lt = loss_test(Z_test, y_test, beta_mm[:,k], grpsizes, indxs, ntest,Rs,p)   
    append!( lt_mm, lt )
end

writedlm("real_data" * ".csv", [cumsum(times_mm) cumsum(times_bcd) cumsum(iters) cumsum(iters_bcd) lamrange], ",")
writedlm("test_error" * ".csv", [lt_bcd lt_mm] , ",")