
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
    function leftcanonical!(
        MPS::AbstractVector{<:AbstractArray{<:Number, 3}};
        Nkeep::Int=typemax(Int), tolerance::Float64=0.0
    )
Bring the MPS into left canonical form. This method transforms the MPS in-place, i.e.
overwrites data. If you want to keep the original MPS, use `leftcanonical()` instead.

Note that to apply this method to subsets of the MPS, you have to use
`view(MPS, start:end)`.
"""
function leftcanonical!(
    MPS::AbstractVector{<:AbstractArray{<:Number, 3}};
    Nkeep::Int=typemax(Int), tolerance::Float64=0.0
)
    for siteindex in 1:length(MPS)-1
        A, Lambda = svdleft(MPS[siteindex]; Nkeep=Nkeep, tolerance=tolerance)
        MPS[siteindex] = A
        MPS[siteindex+1] = contract(Lambda, [2], MPS[siteindex+1], [1])
    end
end

"""
    leftcanonical(
        MPS::AbstractVector{<:AbstractArray{<:Number, 3}};
        Nkeep::Int=typemax(Int), tolerance::Float64=0.0
    )
Bring the MPS into left canonical form. This method does not modify the original MPS,
but returns a new one in left canonical form.
"""
function leftcanonical(
    MPS::AbstractVector{<:AbstractArray{<:Number, 3}};
    Nkeep::Int=typemax(Int), tolerance::Float64=0.0
)
    MPScopy = deepcopy(MPS)
    leftcanonical!(MPScopy; Nkeep=Nkeep, tolerance=tolerance)
    return MPScopy
end

"""
    rightcanonical!(
        MPS::AbstractVector{<:AbstractArray{<:Number, 3}};
        Nkeep::Int=typemax(Int), tolerance::Float64=0.0
    )

Bring the MPS into right canonical form. This method transforms the MPS in-place, i.e.
overwrites data. If you want to keep the original MPS, use `rightcanonical()` instead.

Note that to apply this method to subsets of the MPS, you have to use
`view(MPS, start:end)`.
"""
function rightcanonical!(
    MPS::AbstractVector{<:AbstractArray{<:Number, 3}};
    Nkeep::Int=typemax(Int), tolerance::Float64=0.0
)
    for siteindex in length(MPS):-1:2
        Lambda, B = svdright(MPS[siteindex]; Nkeep=Nkeep, tolerance=tolerance)
        MPS[siteindex] = B
        MPS[siteindex-1] = contract(MPS[siteindex-1], [3], Lambda, [1])
    end
end

"""
    rightcanonical(
        MPS::AbstractVector{<:AbstractArray{<:Number, 3}};
        Nkeep::Int=typemax(Int), tolerance::Float64=0.0
    )

Bring the MPS into right canonical form. This method does not modify the original MPS,
but returns a new one in right canonical form.
"""
function rightcanonical(
    MPS::AbstractVector{<:AbstractArray{<:Number, 3}};
    Nkeep::Int=typemax(Int), tolerance::Float64=0.0
)
    MPScopy = deepcopy(MPS)
    rightcanonical!(MPScopy; Nkeep=Nkeep, tolerance=tolerance)
    return MPScopy
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

    leftcanonical!(
        view(MPS, 1:canonicalcenter);
        Nkeep=Nkeep, tolerance=tolerance
    )
    rightcanonical!(
        view(MPS, canonicalcenter:length(MPS));
        Nkeep=Nkeep, tolerance=tolerance
    )
end


"""
Bring the MPS into bond form with respect to the site `canonicalcenter`. 
This means the "bond matrix" sits between site canonicalcenter and canonicalcenter

Output: "MPS" (missing information in Lambda) and the Lambda matrix
"""

function bondcanonical(
    MPS::AbstractVector{<:AbstractArray{<:Number, 3}},
    canonicalcenter::Int=1;
    Nkeep::Int=typemax(Int), tolerance::Float64=0.0
)
    # Check if canonicalcenter is within bounds
    if canonicalcenter < 1 || canonicalcenter > length(MPS)
        error("canonicalcenter out of bounds.")
    end

    MPScopy = deepcopy(MPS)
    sitecanonical!(MPScopy, canonicalcenter; Nkeep=Nkeep, tolerance=tolerance)
    U, S, Vd = svdleftright(contract(MPScopy[canonicalcenter], 3, MPScopy[canonicalcenter+1], 1))
    MPScopy[canonicalcenter]=U
    MPScopy[canonicalcenter+1] = Vd
    return MPScopy, S
end 

"""
    sitecanonical(
        MPS::AbstractVector{<:AbstractArray{<:Number, 3}},
        canonicalcenter::Int=1;
        Nkeep::Int=typemax(Int), tolerance::Float64=0.0
    )
Bring the MPS into canonical form with respect to the site `canonicalcenter`. This method
does not modify the original MPS, but returns a new one in canonical form.
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


## add alternative implementation of leftcanonical, based on thin Qr decomposition for tensors
# this is more compute efficient but doesn't allow truncation

using LinearAlgebra

"""
    left_canonical_qr!(mps::Vector{<:AbstractArray})

Convert `mps` in-place to left-canonical form via skinny (thin) QR factorizations.
Assumes each tensor has dimensions (bond_left, phys_dim, bond_right).
Returns the list of updated bond dimensions.
"""
function left_canonical_qr!(mps::Vector{<:AbstractArray})
    L = length(mps)

    for site in 1:L-1
        A = mps[site]
        bond_left, phys_dim, bond_right = size(A)

        # Merge the left and physical legs into a matrix of size (bond_left*phys_dim) × bond_right
        M = reshape(A, bond_left*phys_dim, bond_right)

        # Thin QR: M = Q * R, with Q size (bond_left*phys_dim)×new_bond
        F = qr(M, Val(true))
        Q = Matrix(F.Q)  # isometric factor
        R = F.R          # small upper-triangular matrix
        new_bond = size(Q, 2)

        # Reshape Q back into the site tensor shape (bond_left, phys_dim, new_bond)
        mps[site] = reshape(Q, bond_left, phys_dim, new_bond)

        # Absorb R into the next site
        B = mps[site+1]  # size (bond_right, phys_dim_next, bond_right_next)

        # contract the matrix R with B
        mps[site+1] = contract(R, 2, B, 1)
    end

end



function canonForm(M,id,Nkeep=Inf)
# < Description >
#
# M,S,dw = canonForm (M,id [,Nkeep])
#
# Obtain the canonical forms of MPS. It brings the tensors M[1], ..., M[id]
# into the left-canonical form & the others M[id+1], ..., M[end] into the
# right-canonical form.
#
# < Input >
# M : [Any array] MPS of length length(M). Each cell element is a rank-3
#       tensor; where the first; second; & third dimensions are
#       associated with left, bottom [i.e., local], & right legs
#       respectively.
# id : [integer] Index for the bond connecting the tensors M[id] &
#       M[id+1]. With respect to the bond, the tensors to the left
#       (right) are brought into the left-(right-)canonical form. If id ==
#       0; the whole MPS will be in the right-canonical form.
#
# < Option >
# Nkeep : [integer] Maximum bond dimension. That is, only Nkeep the
#       singular values & their associated singular vectors are kept at
#       each iteration.
#       (Default: Inf)
#
# < Output >
# M : [Any array] Left-, right-, | bond-canonical form from input M
#       depending on id; as follows:
#       * id == 0: right-canonical form
#       * id == length(M): left-canonical form
#       * otherwise: bond-canonical form
# S : [column vector] Singular values at the bond between M[id] & M[id+1]. 
# dw : [column vector] Vector of length length(M)-1. dw[n] means the
#       discarded weight (i.e., the sum of the square of the singular  
#       values that are discarded) at the bond between M[n] & M[n+1].
#
# Written originally by S.Lee in 2019 in terms of MATLAB.
# Transformed by Changkai Zhang in 2022 into Julia.

dw = zeros(length(M)-1,1); # discarded weights

# # Bring the left part of MPS into the left-canonical form
for it = (1:id)
    
    # reshape M[it] & SVD
    T = M[it]
    T = reshape(T,(size(T,1)*size(T,2),size(T,3)))
    svdT = LinearAlgebra.svd(T)
    U = svdT.U
    S = svdT.S
    V = svdT.Vt'     
    Svec = S; # vector of singular values
    
    # truncate singular values/vectors; keep up to Nkeep. Truncation at the
    # bond between M[id] & M[id+1] is performed later.
    if ~isinf(Nkeep) && (it < id)
        nk = min(length(Svec),Nkeep); # actual number of singular values/vectors to keep
        dw[it] = dw[it] + sum(Svec[nk+1:end].^2); # discarded weights
        U = U[:,(1:nk)]
        V = V[:,(1:nk)]
        Svec = Svec[1:nk]
    end
    
    S = diagm(Svec); # return to square matrix
    
    # reshape U into rank-3 tensor, & replace M[it] with it
    M[it] = reshape(U,(size(U,1)÷size(M[it],2),size(M[it],2),size(U,2)))
    
    if it < id
        # contract S & V' with M[it+1]
        M[it+1] = contract_old(S*V',2,2,M[it+1],3,1)
    else
        # R1: tensor which is the leftover after transforming the left
        #   part. It will be contracted with the counterpart R2 which is
        #   the leftover after transforming the right part. Then R1*R2 will
        #   be SVD-ed & its left/right singular vectors will be
        #   contracted with the neighbouring M-tensors.
        R1 = S*V'
    end
    
end

# # In case of fully right-canonical form; the above for-loop is not executed
if id == 0
    R1 = 1
end
    
# # Bring the right part into the right-canonical form
for it = (length(M):-1:id+1)
    
    # reshape M[it] & SVD
    T = M[it]
    T = reshape(T,(size(T,1),size(T,2)*size(T,3)))
    svdT = LinearAlgebra.svd(T)
    U = svdT.U
    S = svdT.S
    V = svdT.Vt' 
    Svec = S; # vector of singular values
    
    # truncate singular values/vectors; keep up to Nkeep. Truncation at the
    # bond between M[id] & M[id+1] is performed later.
    if ~isinf(Nkeep) && (it > (id+1))
        nk = min(length(Svec),Nkeep); # actual number of singular values/vectors to keep
        dw[it-1] = dw[it-1] + sum(Svec[nk+1:end].^2); # discarded weights
        U = U[:,(1:nk)]
        V = V[:,(1:nk)]
        Svec = Svec[1:nk]
    end
    
    S = diagm(Svec); # return to square matrix
    
    # reshape V' into rank-3 tensor, replace M[it] with it
    M[it] = reshape(V',(size(V,2),size(M[it],2),size(V,1)÷size(M[it],2)))
    
    if it > (id+1)
        # contract U & S with M[it-1]
        M[it-1] = contract_old(M[it-1],3,3,U*S,2,1)
    else
        # R2: tensor which is the leftover after transforming the right
        #   part. See the description of R1 above.
        R2 = U*S
    end
    
end

# # In case of fully left-canonical form; the above for-loop is not executed
if id == length(M)
    R2 = 1
end

# # SVD of R1*R2; & contract the left/right singular vectors to the tensors

    svdT = LinearAlgebra.svd(R1*R2)
    U = svdT.U
    S = svdT.S
    V = svdT.Vt' 

# truncate singular values/vectors; keep up to Nkeep. At the leftmost &
# rightmost legs [dummy legs], there should be no truncation, since they
# are already of size 1.
if ~isinf(Nkeep) && (id > 0) && (id < length(M))
    
    nk = min(length(S),Nkeep); # actual number of singular values/vectors
    dw[id] = dw[id] + sum(S[nk+1:end].^2); # discarded weights
    U = U[:,(1:nk)]
    V = V[:,(1:nk)]
    S = S[1:nk]
    
end

if id == 0 # fully right-canonical form
    # U is a single number which serves as the overall phase factor to the
    # total many-site state. So we can pass over U to V'.
    M[1] = contract_old(U*V',2,2,M[1],3,1)
elseif id == length(M) # fully left-canonical form
    # V' is a single number which serves as the overall phase factor to the
    # total many-site state. So we can pass over V' to U.
    M[end] = contract_old(M[end],3,3,U*V',2,1)
else
    M[id] = contract_old(M[id],3,3,U,2,1)
    M[id+1] = contract_old(V',2,2,M[id+1],3,1)
end

return M,S,dw

end

