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
