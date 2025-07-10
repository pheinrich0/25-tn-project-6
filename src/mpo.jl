
function mpo_expectation(W::Vector{<:AbstractArray{<:Number,4}}, 
                         MPS::Vector{<:AbstractArray{<:Number,3}})
    # Initialize environment as scalar identity in a 3-leg tensor form
    C = ones(eltype(W[1]), (1, 1, 1))

    for i in eachindex(W)
        C = updateLeft(C, MPS[i], W[i], MPS[i])
    end

    # At the end, C is (1,1,1), so extract scalar value
    return real(C[1,1,1])
end

"""
    mpo_transition(W, MPS1, MPS2)

Compute the transition amplitude <s1|MPO|s2> by contracting MPO W
between two different MPSs, MPS1 (bra) and MPS2 (ket).
"""
function mpo_transition(W::Vector{<:AbstractArray{<:Number,4}},
                        MPS1::Vector{<:AbstractArray{<:Number,3}},
                        MPS2::Vector{<:AbstractArray{<:Number,3}})
    # Initialize environment as scalar identity in a 3-leg tensor form
    C = ones(eltype(W[1]), (1, 1, 1))

    for i in eachindex(W)
        # updateLeft with MPS1 on the bra leg and MPS2 on the ket leg
        C = updateLeft(C, MPS2[i], W[i], MPS1[i])
    end

    # Extract the scalar amplitude (could be complex)
    return C[1,1,1]
end


# applies mpo to mps state
# for each site, leg 2 of the mpo is contracted with leg 2 of the mps
function apply_mpo(W, MPS, Ntrunc::Integer)
    # Ensure the MPO and MPS have the same number of sites
    L = length(W)
    if length(MPS) != L
        error("MPO and MPS must have the same number of sites: got $(length(W)) and $(length(MPS))")
    end

    # Check physical dimensions match at each site
    for i in 1:L
        p_mpo = size(W[i], 2)
        p_mps = size(MPS[i], 2)
        if p_mpo != p_mps
            error("Physical dimension mismatch at site $i: MPO has size $p_mpo, MPS has size $p_mps")
        end
    end
    
    result = Vector{Array{ComplexF64, 3}}(undef, L)
# Apply MPO to MPS at each site via contraction
    for i in 1:L
        # Contract physical leg (leg 2) of W[i] with MPS[i]
        siteITensor = contract(MPS[i], 2, W[i], 4, [1, 3 ,4, 2 ,5])
        IdLeft = identity(siteITensor, 1, siteITensor, 2)
        IdRight = identity(siteITensor, 4, siteITensor, 5)
        siteITensor = contract(IdLeft, [1,2], siteITensor, [1,2])
        siteITensor = contract(siteITensor, [3,4], IdRight, [1,2])
        result[i] = siteITensor
    end

    # introduce truncation of the bond dims to Nkeep
    leftcanonical!(result)
    rightcanonical!(result; Nkeep = Ntrunc)
    
    return result
end


function applyMPO(mps::Vector{Array{ComplexF64, 3}}, mpo::Vector{Array{ComplexF64, 4}}, Nkeep::Int=Inf)
    result = Vector{Array{ComplexF64, 3}}(undef, length(mps))
    for i in 1:length(mps)
        mps_new = contract(mps[i], [2], mpo[i], [4])
        # combine legs (1, 3) and (2, 5)
        id14 = identity(mps_new, 1, mps_new, 3)
        id23 = identity(mps_new, 2, mps_new, 5)
        mps_new = contract(mps_new, [1, 3], id14, [1, 2])
        mps_new = contract(mps_new, [1, 3], id23, [1, 2])
        result[i] = permutedims(mps_new, (2, 1, 3))
    end
    #TODO change leftcanonical to a QR decomposition instead of SVD
    leftcanonical!(result)
    rightcanonical!(result; Nkeep=Nkeep)
    return result
end

# Remark: Sefis function doesnt modify the input mpo