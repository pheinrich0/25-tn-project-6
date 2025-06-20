using LinearAlgebra
using JLD2

import tn_julia: getLocalSpace, contract

# Parameters
beta = 20.0                   # total imaginary time
Nsteps = 2000              # number of Trotter steps
tauT = beta / Nsteps            # Trotter time step

# Spin-1/2 local space tensor (rank-3: left-phys-right)
S = getLocalSpace("Spin", 1/2).S
So = S  # odd site tensor
Se = S  # even site tensor

# Construct two-site Hamiltonians via tensor contraction: <S S>
Ho = contract(So, [2], conj(permutedims(Se, (3,2,1))), [2])  # odd bond
He = contract(Se, [2], conj(permutedims(So, (3,2,1))), [2])  # even bond

# Convert to matrix form for exponentiation
dims_Ho = size(Ho)
dims_He = size(He)

Homat = reshape(permutedims(Ho, (1,3,2,4)), dims_Ho[1]*dims_Ho[3], dims_Ho[2]*dims_Ho[4])
Hemat = reshape(permutedims(He, (1,3,2,4)), dims_He[1]*dims_He[3], dims_He[2]*dims_He[4])

# Matrix exponentiation
expHo_mat = exp(-tauT * Homat)
expHe_mat = exp(-tauT * Hemat)

# Reshape back to rank-4 tensors: (left1, left2, right1, right2)
expHo = reshape(expHo_mat, dims_Ho[1], dims_Ho[3], dims_Ho[2], dims_Ho[4])
expHe = reshape(expHe_mat, dims_He[1], dims_He[3], dims_He[2], dims_He[4])

# Save all relevant tensors
@save "problem/set04/iTEBD_i.jld2" So Se Homat Hemat expHo expHe