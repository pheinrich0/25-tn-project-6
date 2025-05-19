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

function svdleft(
    T::AbstractArray{ValueType};
    Nkeep::Int=typemax(Int), tolerance::Float64=0.0
) where {ValueType<:Number}
    U, S, Vd, _ = svd(T, [1], Nkeep=Nkeep, tolerance=tolerance)
    return U * Diagonal(S), Vd
end

function svdright(
    T::AbstractArray{ValueType};
    Nkeep::Int=typemax(Int), tolerance::Float64=0.0
) where {ValueType<:Number}
    U, S, Vd, _ = svd(T, collect(1:ndims(T)-1), Nkeep=Nkeep, tolerance=tolerance)
    return U, Diagonal(S) * Vd
end

function svdleftright(
    T::AbstractArray{ValueType};
    Nkeep::Int=typemax(Int), tolerance::Float64=0.0
) where {ValueType<:Number}
    U, S, Vd, _ = svd(T, [1, 2], Nkeep=Nkeep, tolerance=tolerance)
    return U, Diagonal(S), Vd
end
