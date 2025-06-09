
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
