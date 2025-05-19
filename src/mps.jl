
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

function canonicalform(MPS::Array{<:AbstractArray{<:Number}, 3}; Nkeep::Int=typemax(Int), tolerance::Float64=0.0)
    # Convert MPS to canonical form
    for i in 1:length(MPS)-1
        U, S, Vd, w = tn_julia.svdleftright(MPS[i], Nkeep=Nkeep, tolerance=tolerance)
        MPS[i] = U
        MPS[i+1] = contract(MPS[i], [3], Vd, [1])
    end
    return MPS
end
