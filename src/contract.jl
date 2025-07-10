"""
    contract(A, Aindices, B, Bindices, indexpermutation=[])

Contract tensors `A` and `B`.
The legs to be contracted are given by `Aindices` and `Bindices`.

Input:
- `A`, `B` (numeric arrays): the tensors to be contracted.
- `Aindices`, `Bindices` (ordered collection of integers):
    Indices for the legs of `A` and `B` to be contracted. The `Aindices[n]`-th leg of `A`
    will be contracted with the `Bindices[n]`-th leg of `B`. `Aindices` and `Bindices`
    should have the same size. If they are both empty, the result will be the direct product
    of `A` and `B`.
- `indexpermutation` (ordered collection of integers):
    Permutation of of the output tensor's indices to be performed after contraction.

Output:
- Contraction of `A` and `B`. All non-contracted legs are in the following order: first, all
    legs previously attached to `A`, in order, then all legs previously attached to `B`, in
    order.
"""
function contract(A, Aindices, B, Bindices, indexpermutation=[])
    if length(Aindices) != length(Bindices)
        error("Different number of legs to contract for tensors A and B.")
    end
    if any(Aindices .< 1) || any(Aindices .> ndims(A))
        error("Got indices out of range for tensor A in contract().")
    end
    if any(Bindices .< 1) || any(Bindices .> ndims(B))
        error("Got indices out of range for tensor B in contract().")
    end

    # indices of legs *not* to be contracted
    Arestinds = setdiff(1:ndims(A), Aindices)
    Brestinds = setdiff(1:ndims(B), Bindices)

    # reshape tensors into matrices with "thick" legs
    A2 = reshape(
        permutedims(A, tuple(cat(dims=1, Arestinds, Aindices))...),
        (prod(size(A)[Arestinds]), prod(size(A)[Aindices]))
    )
    B2 = reshape(
        permutedims(B, tuple(cat(dims=1, Bindices, Brestinds))...),
        (prod(size(B)[Bindices]), prod(size(B)[Brestinds]))
    )
    C2 = A2 * B2 # matrix multiplication

    # size of C
    Cdim = (size(A)[Arestinds]..., size(B)[Brestinds]...)
    # reshape matrix to tensor
    C = reshape(C2, Cdim)

    if !isempty(indexpermutation) # if permutation option is given
        C = permutedims(C, indexpermutation)
    end

    return C
end


function contract_old(A,rankA,idA,B,rankB,idB,idC=[])
# < Description >
#
# C = contract(A,rankA,idA,B,rankB,idB [,idC]) 
#
# Contract tensors A & B. The legs to be contracted are given by idA
# & idB.
#
# < Input >
# A, B : [numeric array] Tensors.
# rankA, rankB : [integer] Rank of tensors. Since MATLAB removes the last
#       trailing singleton dimensions; it is necessary to set rankA &
#       rankB not to miss the legs of size 1 (or bond dimension 1
#       equivalently).
# idA, idB : [integer vector] Indices for the legs of A & B to be
#        contracted. The idA[n]-th leg of A & the idB[n]-th leg of B will
#        be contracted, for all 1 <= n <= length(idA). idA & idB should
#        have the same number of elements. If they are both empty; C will
#        be given by the direct product of A & B.
# 
# < Option >
# idC : [integer tuple] To permute the resulting tensor after contraction
#       assign the permutation indices as idC. If the dummy legs are
#       attached [see the description of C below], this permutation is
#       applied *after* the attachment.
#       (Default: no permutation)
#
# < Output >
# C : [numeric array] Contraction of A & B. If idC is given, the
#       contracted tensor is permuted accordingly. If the number of open
#       legs are smaller than 2; the dummy legs are assigned to make the
#       result array C be two-dimensional.
#
# Written originally by S.Lee in 2017 in terms of MATLAB.
# Transformed by Changkai Zhang in 2022 into Julia.

# # check the integrity of input & option
Asz = size(A); Bsz = size(B); # size of Tensors

if length(Asz) != rankA
    error("ERR: Input tensor A has a different rank from input rankA.")
end

if length(Bsz) != rankB
    error("ERR: Input tensor B has a different rank from input rankB.")
end

if length(idA) != length(idB)
    error("ERR: Different # of leg indices to contract for tensors A & B.")
end

# # # # Main computational part [start] # # # #
# indices of legs *not* to be contracted
idA2 = setdiff(1:rankA,idA); 
idB2 = setdiff(1:rankB,idB);

# reshape tensors into matrices with "thick" legs
A2 = reshape(permutedims(A,tuple(cat(dims=1,idA2,idA))...),(prod(Asz[idA2]),prod(Asz[idA]))); # note: prod([]) .== 1
B2 = reshape(permutedims(B,tuple(cat(dims=1,idB,idB2))...),(prod(Bsz[idB]),prod(Bsz[idB2])))
C2 = A2*B2; # matrix multiplication

# size of C
if (length(idA2) + length(idB2)) > 1
    Cdim = (Asz[idA2]...,Bsz[idB2]...)
else
    # place dummy legs x of singleton dimension when all the legs of A (or
    # B) are contracted with the legs of B [or A]
    Cdim = 1
end

# reshape matrix to tensor
C = reshape(C2,Cdim)

if ~isempty(idC) # if permutation option is given
    C = permutedims(C,idC)
end
# # # # Main computational part [end] # # # #

return C

end

