using LinearAlgebra

"""
    identity(T, leg::Int)

Construct the identity operator in the space of a given leg of a tensor.
"""
function identity(T, leg::Int)
    return I(size(T, leg))
end

"""
    identity(A, legA::Int, B, legB::Int)

Construct the identity operator in the direct product space of two tensor legs.
E.g. consider two tensors A, B; this function constructs the identity operator I that takes
the following place:
    --A--I--
      |  |
         B
         |
"""
function identity(A, legA::Int, B, legB::Int)
    targetsize = size(A, legA) * size(B, legB)
    return reshape(
        I(targetsize),
        (size(A, legA), size(B, legB), targetsize)
    )
end

