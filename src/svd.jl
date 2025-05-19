"""
    svd(
        T::AbstractArray{ValueType}, indicesU::AbstractVector{Int};
        Nkeep::Int=typemax(Int), tolerance::Float64=0.0
    )

Computes the SVD of tensor T, with arbitrary assignment of legs to U and V.

Input:
- `T`: the tensor to be decomposed.
- `indicesU`: legs of T to be assigned to U.
- `Nkeep`: maximum number of singular values to keep. Default is `typemax(Int)`.
- `tolerance`: minimum magnitude of singular value to keep. Default is `0.0`.

Output:
- `U`: the left singular vectors of T, with legs given in `indicesU`.
- `S`: the singular values of T, in decreasing order.
- `Vd`: the right singular vectors of T.
- `discardedweight`: the sum of squares of the singular values that were discarded.
"""
function svd(
    T::AbstractArray{ValueType}, indicesU::AbstractVector{Int};
    Nkeep::Int=typemax(Int), tolerance::Float64=0.0
) where {ValueType<:Number}
    if isempty(T)
        return (zeros(0, 0), zeros(0), zeros(0, 0), 0.0)
    end

    indicesV = setdiff(1:ndims(T), indicesU)
    Tmatrix = reshape(
        permutedims(T, cat(dims=1, indicesU, indicesV)),
        (prod(size(T)[indicesU]), prod(size(T)[indicesV]))
    )
    svdT = LinearAlgebra.svd(Tmatrix)

    Nkeep = min(Nkeep, size(svdT.S, 1))
    Ntolerance = findfirst(svdT.S .< tolerance)
    if !isnothing(Ntolerance)
        Nkeep = min(Nkeep, Ntolerance - 1)
    end

    U = reshape(svdT.U[:, 1:Nkeep], size(T)[indicesU]..., Nkeep)
    S = svdT.S[1:Nkeep]
    Vd = reshape(svdT.Vt[1:Nkeep, :], Nkeep, size(T)[indicesV]...)
    discardedweight = sum(svdT.S[Nkeep+1:end] .^ 2)

    return (U, S, Vd, discardedweight)
end

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
    MPS = Array{ValueType, 3}[] # Initialize an empty vector to store the MPS tensors
    discardedweight = Float64[]
    while ndims(tail) > 2
        U, S, tail, w = tn_julia.svd(tail, [1, 2]; Nkeep=Nkeep, tolerance=tolerance)
        M = contract(U, [3], Diagonal(S), [1]) # Absorb S to the left: M = U * S
        push!(MPS, M)
        push!(discardedweight, w)
    end
    return MPS, discardedweight
end
