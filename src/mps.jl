
"""
    tensor2MPS(T::AbstractArray{ValueType})

Factorizes a tensor `T` into a matrix product state (MPS) using SVD.

Input:
- `T`: the tensor to be factorized.
- `Nkeep`: maximum number of singular values to keep. Default is `typemax(Int)`.
- `tolerance`: minimum magnitude of singular value to keep. Default is `0.0`.
Output:
- `MPS`: an array of tensors representing the MPS.
- `discardedweight`: the sum of squares of the singular values that were discarded, for each bond.
"""
function tensor2MPS(
    T::AbstractArray{ValueType};
    Nkeep::Int=typemax(Int), tolerance::Float64=0.0
) where {ValueType<:Number}
    # Dummy legs added for convenient reshaping into MPS
    tail = reshape(T, (1, size(T)..., 1))
    MPS = Array{ValueType,3}[] # Initialize an empty vector to store the MPS tensors
    discardedweight = Float64[]
    while ndims(tail) > 2
        U, S, tail, w = tn_julia.svd(tail, [1, 2]; Nkeep=Nkeep, tolerance=tolerance)
        M = contract(U, [3], Diagonal(S), [1]) # Absorb S to the left: M = U * S
        push!(MPS, M)
        push!(discardedweight, w)
    end
    return MPS, discardedweight
end

"""
    sitecanonical!(
        MPS::AbstractVector{<:AbstractArray{<:Number, 3}},
        canonicalcenter::Int=1;
        Nkeep::Int=typemax(Int), tolerance::Float64=0.0
    )

Bring the MPS into canonical form with respect to the site `canonicalcenter`. This method
transforms the MPS in-place, i.e. overwrites data. If you want to keep the original MPS,
use `sitecanonical()` instead.

Input:
- `MPS`: the MPS to be transformed.
- `canonicalcenter`: the site to be brought into canonical form. Default is 1.
- `Nkeep`: maximum number of singular values to keep. Default is `typemax(Int)`.
- `tolerance`: minimum magnitude of singular value to keep. Default is `0.0`.
"""
function sitecanonical!(
    MPS::AbstractVector{<:AbstractArray{<:Number, 3}},
    canonicalcenter::Int=1;
    Nkeep::Int=typemax(Int), tolerance::Float64=0.0
)
    # Check if canonicalcenter is within bounds
    if canonicalcenter < 1 || canonicalcenter > length(MPS)
        error("canonicalcenter out of bounds.")
    end

    for siteindex in 1:canonicalcenter-1
        A, Lambda = svdleft(MPS[siteindex]; Nkeep=Nkeep, tolerance=tolerance)
        MPS[siteindex] = A
        MPS[siteindex+1] = contract(Lambda[2], MPS[siteindex+1], [1])
    end

    for siteindex in length(MPS):-1:canonicalcenter+1
        Lambda, B = svdright(MPS[siteindex]; Nkeep=Nkeep, tolerance=tolerance)
        MPS[siteindex] = B
        MPS[siteindex-1] = contract(MPS[siteindex-1], [3], Lambda, [1])
    end
end

"""
    sitecanonical(
        MPS::AbstractVector{<:AbstractArray{<:Number, 3}},
        canonicalcenter::Int=1;
        Nkeep::Int=typemax(Int), tolerance::Float64=0.0
    )
Bring the MPS into canonical form with respect to the site `canonicalcenter`. This method
does not modify the original MPS, but returns a new one in canonical form.

Input:
- `MPS`: the MPS to be transformed.
- `canonicalcenter`: the site to be brought into canonical form. Default is 1.
- `Nkeep`: maximum number of singular values to keep. Default is `typemax(Int)`.
- `tolerance`: minimum magnitude of singular value to keep. Default is `0.0`.
Output:
- A new MPS in canonical form with respect to the site `canonicalcenter`.
"""
function sitecanonical(
    MPS::AbstractVector{<:AbstractArray{<:Number, 3}},
    canonicalcenter::Int=1;
    Nkeep::Int=typemax(Int), tolerance::Float64=0.0
)
    MPScopy = deepcopy(MPS)
    sitecanonical!(MPScopy, canonicalcenter; Nkeep=Nkeep, tolerance=tolerance)
    return MPScopy
end
