## functions to bring the mps into schmidt decomposition / bond canonical form
# -> can be found in
# and calculate the entanglement entropy.

function entanglement_entropy(
    MPS::AbstractVector{<:AbstractArray{<:Number, 3}},
    site::Int
)
    # Call bondcanonical to get S matrix at the bond
    # _, S = tn.bondcanonical(MPS, site) 
    # use provided fct instead
    _, s, _ = tn.canonForm(MPS,site)

    # Get the singular values as a vector
    #s = diag(S)

    # Normalize to make it a proper probability distribution (Schmidt weights)
    s_norm = s / sum(s)

    # Compute von Neumann entropy: -∑ p log(p)
    entropy = -sum(p -> p > 0 ? p * log(p) : 0.0, s_norm)

    return entropy
end

