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