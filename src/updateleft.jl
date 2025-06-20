"""
    updateLeft(C, B, X, A)

Obtain the operator Cleft that acts on the Hilbert space to the left of an MPS site through
contraction, in the following pattern,
```
    .-A-- 3
    | |
    C-X-- 2
    | |
    '-B*- 1
```
where B* is the complex conjugate of B. Note the leg ordering (bottom-to-top).

Input:
- `C` (3-leg tensor): environment to the left of the current site.
- `B`, `A` (3-leg tensors): ket tensors on the current site.
- `X` (4-leg tensor): local operator. # ordering: left bottom right top

Output:
- Tensor corresponding to the fully contracted tensor network shown above.
"""
function updateLeft(C, B, X, A)
# < Description >
#
# Cleft = updateLeft(Cleft,rankC,B,X,rankX,A)
#
# Contract the operator Cleft that act on the Hilbert space of the left
# part of the MPS [i.e., left of a given site] with the tensors B, X, &
# A; acting on the given site.
#
# < Input >
# Cleft : [tensor] Rank-2 | 3 tensor from the left part of the system. If
#       given as empty [], then Cleft is considered as the identity tensor
#       of rank 2 [for rank(X) .< 4] | rank 3 [for rank(X) .== 4].
# rankC : [integer] Rank of Cleft.
# B, A : [tensors] Ket tensors, whose legs are ordered as left - bottom
#       (local physical) - right. In the contraction, the Hermitian
#       conjugate [i.e., bra form] of B is used, while A is contracted as
#       it is. This convention of inputting B as a ket tensor reduces extra
#       computational cost of taking the Hermitian conjugate of B.
# X : [tensor] Local operator with rank 2 | 3. If given as empty [], then
#       X is considered as the identity.
# rankX : [integer] Rank of X.
#
# < Output >
# Cleft : [tensor] Contracted tensor. The tensor network diagrams
#       describing the contraction are as follows.
#       * When Cleft is rank-3 & X is rank-2:
#                    1     3
#          /--------->- A ->--            /---->-- 3
#          |            | 2               |
#        3 ^            ^                 |
#          |    2       | 2               |      
#        Cleft---       X         =>    Cleft ---- 2
#          |            | 1               |
#        1 ^            ^                 |
#          |            | 2               |
#          \---------<- B'-<--            \----<-- 1
#                    3     1
#       * When Cleft is rank-2 & X is rank-3:
#                    1     3
#          /--------->- A ->--            /---->-- 3
#          |            | 2               |
#        2 ^            ^                 |
#          |          3 |   2             |      
#        Cleft          X ----    =>    Cleft ---- 2
#          |          1 |                 |
#        1 ^            ^                 |
#          |            | 2               |
#          \---------<- B'-<--            \----<-- 1
#                    3     1
#       * When both Cleft & X are rank-3:
#                    1     3
#          /--------->- A ->--            /---->-- 2
#          |            | 2               |
#        3 ^            ^                 |
#          |   2     2  | 3               |      
#        Cleft--------- X         =>    Cleft
#          |            | 1               |
#        1 ^            ^                 |
#          |            | 2               |
#          \---------<- B'-<--            \----<-- 1
#                    3     1
#       * When Cleft is rank3 & X is rank-4:
#                    1     3
#          /--------->- A ->--            /---->-- 3
#          |            | 2               |
#        3 ^            ^                 |
#          |   2    1   | 4               |      
#        Cleft--------- X ---- 3   =>   Cleft ---- 2
#          |            | 2               |
#        1 ^            ^                 |
#          |            | 2               |
#          \---------<- B'-<--            \----<-- 1
#                    3     1
#       Here B' denotes the Hermitian conjugate [i.e., complex conjugate
#       & permute legs by [3 2 1]] of B.
#
# Written by H.Tu [May 3,2017]; edited by S.Lee [May 19,2017]
# Rewritten by S.Lee [May 5,2019]
# Updated by S.Lee [May 27,2019]: Case of rank-3 Cleft & rank-4 X is()
#       added.
# Updated by S.Lee [Jul.28,2020]: Minor fix for the case when Cleft .== []
#       & rank(X) .== 4.
# Transformed by Changkai Zhang into Julia [May 4, 2022]

    # error checking
    if ndims(C) != 3
        error("In updateLeft, got parameter C with $(ndims(C)) dimensions. C must have 3 dimensions.")
    end
    if ndims(B) != 3
        error("In updateLeft, got parameter B with $(ndims(B)) dimensions. B must have 3 dimensions.")
    end
    if ndims(X) != 4
        error("In updateLeft, got parameter X with $(ndims(X)) dimensions. X must have 4 dimensions.")
    end
    if ndims(A) != 3
        error("In updateLeft, got parameter A with $(ndims(A)) dimensions. A must have 3 dimensions.")
    end

    CA = contract(C, [3], A, [1])
    CAX = contract(CA, [2, 3], X, [1, 4])
    return contract(conj(B), [1, 2], CAX, [1, 3], [1, 3, 2])
end

using LinearAlgebra

import tn_julia: contract 

function updateLeft(Cleft, rankC, B, X, rankX, A)
    # error checking
    if !isempty(Cleft) && !(rankC in [2, 3])
        error("ERR: Rank of Cleft or Cright should be 2 | 3.")
    end
    if !isempty(X) && !(rankX in (2:4))
        error("ERR: Rank of X should be 2, 3, | 4.")
    end

    B = conj(B)  # bra = conj(ket), no permute

    if !isempty(X)
        # Apply operator X to A on the physical leg
        T = contract(X, [rankX], A, [2])

        if !isempty(Cleft)
            if (rankC > 2) && (rankX > 2)
                if rankX == 4
                    # contract 2nd and last legs of Cleft with 1st and last of T
                    T = contract(Cleft, [2, rankC], T, [1, rankX])
                else
                    # contract operator-style legs
                    T = contract(Cleft, [2, rankC], T, [2, rankX])
                end
                Cleft = contract(B, [1, 2], T, [1, 2])
            else
                T = contract(Cleft, [rankC], T, [rankX])
                Cleft = contract(B, [1, 2], T, [1, rankC])
            end
        elseif (rankX == 4) && (size(X, 1) == 1)
            # special case: MPO with dummy left leg
            Cleft = contract(B, [1, 2], T, [rankX, 2], [1, 3, 4, 2])
        else
            Cleft = contract(B, [1, 2], T, [rankX, 1])
        end
    elseif !isempty(Cleft)
        T = contract(Cleft, [rankC], A, [1])
        Cleft = contract(B, [1, 2], T, [1, rankC])
    else
        Cleft = contract(B, [1, 2], A, [1, 2])
    end

    return Cleft
end