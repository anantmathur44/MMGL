


using JLD,LinearAlgebra, DelimitedFiles, Roots, Statistics, Distributions, IterativeSolvers, LinearMaps, Random, JLD

function gen_cov_matrix(m,grpsize,ro,psi)    
    bsize = m*grpsize
    covmatrix = zeros(bsize,bsize)
    for i in 1:bsize
        for j in 1:bsize
           if i == j 
               covmatrix[i,j] = 1
           elseif ceil(i/grpsize) == ceil(j/grpsize)
               covmatrix[i,j] = ro
           else
               covmatrix[i,j] = psi
           end
        end
    end
    return covmatrix 
end

function simulate_data_gl(n,grpsize,m,SNR,sigma,k)
    
    Z = Array{Matrix{Float64}}(undef, m)
    mean = zeros(grpsize*m)
    mvn = MvNormal(mean, sigma)
    Zall = rand(mvn, n)'
    thetasall = zeros(1,grpsize*(m))
    thetas = Array{Matrix{Float64}}(undef, m)
    indic =  Array{Integer}(undef, m)
    indic = [zeros(Int, m - k); ones(Int, k)]
    shuffle!(indic)
    for i = 1:m
         theta  = [-2,-1,0,1,2]
         thetas[i] = indic[i]* theta[:,:]
         thetasall[(i-1)*grpsize+1:i*grpsize] = thetas[i]
         Z[i] =  Zall[:,(i-1)*grpsize+1:i*grpsize]
    end
    Zall = reduce(hcat, Z)
    y = zeros(n,1)    
    for i = 1:m
        y += Z[i]*thetas[i]
    end
    noise = thetasall*(sigma*(thetasall'))/SNR
    y = y + sqrt(noise[1])*randn(n)
    y = vec(y)
    return Z, Zall,thetas,y,thetasall,indic
end


function theta_to_beta(theta,Rs,m,grpsize)
    beta = zeros(grpsize*m,1)
    for j = 1:m
        beta[(j-1)*grpsize+1:j*grpsize] = Rs[j]\theta[(j-1)*grpsize+1:j*grpsize] 
    end
    return beta
end

function loss(Z,y,htheta, grpsize,m,n, lambda)
   Zall = reduce(hcat, Z);
   return ((0.5*norm(y- Zall*htheta)^2) + sqrt(grpsize)*lambda*sum([norm(htheta[(j-1)*grpsize+1:j*grpsize]) for j = 1:m]))/n
end


function deviance(Z,y,htheta, grpsize,m,n, lambda)
   Zall = reduce(hcat, Z);
   return ((0.5*norm(y- Zall*htheta)^2))/n
end


function soft_thresh(x,l)
    cons = max(0, 1-(l/abs(x)))
    return x*cons
end

#Norm of diff in theta/grpsize
function group_lasso_mm(
    y::AbstractVector{T}, 
    Zall::AbstractMatrix{T};
    verbose::Bool = false, 
    maxiter::Integer = 5000, 
    funtol::Number = 1e-8,
    gamma::Vector{T} = zeros(T, m),    
    htheta::Vector{T} = zeros(T, sum(grpsizes)),
    m::Number,
    lambdas::AbstractVector{T},
    indxs::Dict,
    grpsizes::Vector{Int}
) where T <: LinearAlgebra.BlasFloat

    n = length(y)
    max_grpsize = maximum(grpsizes)

    # Preallocate working arrays
    r = copy(y)
    hthetaold = zeros(T, length(htheta))
    dd = zeros(T, max_grpsize)  # Adjusted size to handle max rank
    lamrange = length(lambdas)
    iterations = zeros(Int, lamrange)
    actives = zeros(Int, lamrange)
    thetall = zeros(T, length(htheta), lamrange)
    gammall = zeros(T, m, lamrange)
    elapsed_times = Float64[]

    theta = zeros(T, m)
    Gy = copy(y)
    ZGys = zeros(T, sum(grpsizes))
    X_active = zeros(T, n, 0)
    gamma_active = Float64[]
    active_set = Int[]
    group_lengths = Int[]
    
    
    for lami in 1:lamrange
        lambda = lambdas[lami]
        if verbose
            println("Index: $lami")
            println("LAMBDA: $lambda")
        end
        elapsed_time = @elapsed begin
            niter = maxiter  # Initialize niter here
            for iter in 1:maxiter
                dif = 0
                copy!(hthetaold, htheta)
                active_count = length(active_set)
                if verbose
                    #println("Iter: $iter, Active: $active_count")
                end
                
                mul!(ZGys, Zall', Gy)  # In-place multiplication
                for i in 1:m
                    idx_range = indxs[i]
                    grpsize_i = grpsizes[i]
                    theta[i] = norm(ZGys[idx_range])
                end

                alpha = sqrt.(grpsizes) .* lambda
                w = getWeights(alpha, theta, gamma, m)

                for j in 1:m
                    idx_range = indxs[j]
                    grpsize_j = grpsizes[j]
                    if (w[j] + gamma[j]) == 0
                        continue
                    end
                    gammajnew = soft_thresh((w[j] + gamma[j]) * theta[j], w[j] * alpha[j]) / alpha[j]
                    if gammajnew != gamma[j]
                        if gammajnew == 0
                            jr = findfirst(x -> x == j, active_set)
                            if jr !== nothing
                                # Find the length of the group to remove
                                length_to_remove = group_lengths[jr]
                                # Remove the group from `active_set`
                                splice!(active_set, jr)
                                # Remove the corresponding length from `group_lengths`
                                splice!(group_lengths, jr)
                                # Calculate the starting index to remove
                                start_index = sum(group_lengths[1:jr-1]) + 1
                                # Calculate the ending index to remove
                                end_index = start_index + length_to_remove - 1
                                # Remove the corresponding columns from X_active
                                X_active = X_active[:, setdiff(1:end, start_index:end_index)]
                                # Remove the corresponding entries from gamma_active
                                gamma_active = gamma_active[setdiff(1:end,start_index:end_index)]
                            end
                        elseif !(j in active_set)
                            push!(active_set, j)
                            # Get the length of the new group
                            group_length = grpsize_j
                            push!(group_lengths, group_length)
                            # Append new values to gamma_active
                            append!(gamma_active, gammajnew * ones(T, group_length))
                            # Add the corresponding columns to X_active
                            X_active = hcat(X_active, Zall[:, indxs[j]])
                        else
                            #println(active_set)
                            #println(j)
                            jr = findfirst(x -> x == j, active_set)
                            #println(jr)
                            #println(gamma_active)
                            # Update the value in gamma_active for the entire group
                            start_index = sum(group_lengths[1:jr-1]) + 1
                            end_index = start_index + group_lengths[jr] - 1
                            gamma_active[start_index:end_index] .= gammajnew
                        end
                        gamma[j] = gammajnew                    
                        htheta[idx_range] .= gamma[j] .* ZGys[idx_range]
                        dd[1:grpsize_j] .= htheta[idx_range] .- hthetaold[idx_range]
                        dif = max(dif, (1/(grpsize_j)) * (norm(dd[1:grpsize_j])))
                    end
                end

                lm = v -> v + X_active * (gamma_active .* (X_active' * v))
                LM = LinearMap{T}(lm, n, n)
                cg!(Gy, LM, y, maxiter = 4)
                #println(dif)
                if dif < funtol
                    niter = iter
                    #println(loss2(Zall,y,htheta, grpsize,m,n, lambda))
                    break
                end
            end
        end

        push!(elapsed_times, elapsed_time)
        thetall[:, lami] = htheta
        gammall[:, lami] = gamma
        iterations[lami] = niter
        actives[lami] = length(active_set)
        if verbose
            println("Iterations: $niter")
            println("Active: $(actives[lami])")
            println("-------------------------")
        end
    end

    return iterations, actives, thetall, elapsed_times, gammall
end


function group_lasso_bcd(
    y::AbstractVector{T}, 
    Z::Vector{<:AbstractMatrix{T}},
    Zall::AbstractMatrix{T};
    verbose::Bool = false, 
    maxiter::Integer = 5000, 
    funtol::Number = 1e-8,
    htheta::Vector{T} = zeros(T, sum(grpsizes)),
    lambdas::AbstractVector{T},
    indxs::Dict,
    grpsizes::Vector{Int}
) where T <: LinearAlgebra.BlasFloat

    n = length(y)
    m = length(Z)  # Number of variance components

    # Preallocate working arrays
    r = copy(y)
    p = sum(grpsizes)
    hthetaold = zeros(T, p)
    max_grpsize = maximum(grpsizes)
    Zjr = zeros(T, max_grpsize)  # Adjusted size to handle max rank
    thetaj = zeros(T, max_grpsize)  # Adjusted size to handle max rank
    dd = zeros(T, max_grpsize)  # Adjusted size to handle max rank
    diffv = zeros(T, max_grpsize)  # Adjusted size to handle max rank
    temp_r = zeros(T, n)  # Temporary array for Z[j] * htheta and Z[j] * diffv
    lamrange = length(lambdas)
    iterations = zeros(Int, lamrange)
    actives = zeros(Int, lamrange)
    thetall = zeros(T, p, lamrange)
    elapsed_times = Float64[]

    # Extract starting indices for each group from indxs
    start_indices = [indxs[j].start for j in 1:m]

    for lami in 1:lamrange
        lambda = lambdas[lami]
        if verbose
            println("Index: $lami")
            println("LAMBDA: $lambda")
        end
        elapsed_time = @elapsed begin
            niter = maxiter  # Initialize niter here
            for iter in 1:maxiter
                dif = 0
                copy!(hthetaold, htheta)
                if verbose
                    active_count = sum(abs.(htheta[start_indices]) .> 0)
                    #println("Iter: $iter, Active: $active_count")
                end

                for j in 1:m
                    idx_range = indxs[j]
                    grpsize_j = grpsizes[j]

                    # In-place multiplication with variable size
                    mul!(view(Zjr, 1:grpsize_j), Z[j]', r)
                    Zjr[1:grpsize_j] .+= view(htheta, idx_range)

                    norm_Zjr = norm(view(Zjr, 1:grpsize_j))
                    term = 1 - ((sqrt(grpsize_j) * lambda) / norm_Zjr)
                    if term <= 0 
                        thetaj[1:grpsize_j] .= 0
                        if !iszero(view(htheta, idx_range))
                            mul!(temp_r, Z[j], view(htheta, idx_range))
                            r .+= temp_r
                        end
                    else
                        thetaj[1:grpsize_j] .= term .* view(Zjr, 1:grpsize_j)
                        diffv[1:grpsize_j] .= thetaj[1:grpsize_j] .- view(htheta, idx_range)
                        mul!(temp_r, Z[j], view(diffv, 1:grpsize_j))
                        r .-= temp_r
                    end
                    htheta[idx_range] .= thetaj[1:grpsize_j]
                    dd[1:grpsize_j] .= view(htheta, idx_range) .- view(hthetaold, idx_range)
                    dif = max(dif, (1/(grpsize_j)) * (norm(dd[1:grpsize_j])))
                end

                
                if dif < funtol
                    niter = iter
                    break
                end
            end
        end

        push!(elapsed_times, elapsed_time)
        thetall[:, lami] = htheta
        iterations[lami] = niter
        actives[lami] = sum(abs.(htheta[start_indices]) .> 0)
        if verbose
            println("Iterations: $niter")
            println("Active: $(actives[lami])")
            println("-------------------------")
        end
    end

    return iterations, actives, thetall, elapsed_times
end

function  getWeights(alpha, theta, gammat,J)
    d = alpha - theta;
    dsqr = d.^2;
    gb = gammat.*theta;
    idx = .~((d.>= 0).*(gammat .== 0));
    d = d[idx];
    dsqr = dsqr[idx];
    gb = gb[idx];
    a = sum(idx);
    s = sortperm(dsqr, rev = true);
    w = zeros(a,1); 
    for k = 1:a
        v = dsqr[s[k]];
        if k > 1
            w[s[1:k-1]] = (gb[s[1:k-1]])./sqrt(v);
        end
        w[s[k]] = 1-sum(w[s[1:k-1]]);
        c = (gb[s[k]])/d[s[k]];
        if (0 < w[s[k]]) & ((w[s[k]] < c) | (d[s[k]] <= 0))
            break
        end
        v = sum(gb[s[1:k]])^2;
        if (dsqr[s[k]] >= v) & ((v >= dsqr[s[min(J,k+1)]]) | (k == a))
            w[s[1:k]] = gb[s[1:k]]./sqrt(v);
            break
        end
    end

    wfinal = zeros(J,1);
    wfinal[idx] = w;
    return wfinal
end



function center_columns_matrix_array(Z_array::Vector{Matrix{Float64}})
    Z_centered_array = [center_columns(Z) for Z in Z_array]
    return Z_centered_array
end

function center_columns(Z::Matrix)
    col_means = mean(Z, dims=1)
    Z_centered = Z .- col_means
    return Z_centered
end




