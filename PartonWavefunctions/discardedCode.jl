## further test with own apply MPO function
MPS_iter2 = Array{Array{ComplexF64, 3}, 2}(undef, 2 * length(momentaN32), 2N)
mps = [ComplexF64.(T) for T in deepcopy(fermionic_mps)]   # start with the Jordan-Wigner MPS, make sure its datatype is correct

for itN in 1:7
    for alpha in [1/2, -1/2]  # alpha = 0 for spin up, 1 for spin down
        mpo_dk_k = dm_alpha_mpo(itN, alpha, N, A)
        mps = tn.apply_mpo(mpo_dk_k, mps, Dmax)
        if alpha==1/2   # spin up -> odd row index
        println("Iteration $alpha for k = $itN completed." )
        result = MPS_iter[(2*itN-1), 1] ≈ mps[1]
        println("Test result for (2*itN-1), 1: ", result)
        MPS_iter2[(2*itN-1), :] = mps
        else    # spin down -> even row index
            MPS_iter2[2*itN, :] = mps
            println("Iteration $alpha for k = $itN completed." )
            result = MPS_iter[(2*itN), 2] ≈ mps[2]
            println("Test result for (2*itN), 2: ", result)
        end

    end
end
# starting with k=4 the result start disagreeing with the other applyMPO function
check_occupation(MPS_iter[14,:])
mpsvar = deepcopy(MPS_iter2[14,:])
check_occupation(mpsvar) # the occupation number for this version is also slightly lower

for i in eachindex(mpsvar)
    if any(isnan, mpsvar[i]) || any(isinf, mpsvar[i])
        @warn "NaN or Inf in tensor at site $i"
    end
end

for (i, T) in enumerate(mpsvar)
    @info "Site $i norm: ", norm(T)
end

tn.apply_mpo(dm_alpha_mpo(8, 1/2, N, A),mpsvar, Dmax)
tn.applyMPO( mpsvar, dm_alpha_mpo(8, 1/2, N, A), Dmax)
# Conclusion: No fucking idea


v = collect(1:16)
function index_transform(n::Int)
    return v[ceil(Int, n/2)], 1 - mod(n, 2)    
end

index_list = [index_transform(n) for n in 1:32]

momentaN32