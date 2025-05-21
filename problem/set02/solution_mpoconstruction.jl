using LinearAlgebra
import tn_julia: xychainmpo, iterativediagonalization, extendhamiltonian_xy, updateLeft

# part (i): Construct the MPO; see src/xychain.jl
J = 1.0
L = 50
mpo = xychainmpo(L, J)

# part (ii) (1): Compute the ground state using iterative diagonalization
Nkeep = 300
extendhamiltonian(H::AbstractMatrix, A::AbstractArray{<:Number, 3}) = extendhamiltonian_xy(H, A; J=J)
energies, mps = iterativediagonalization(extendhamiltonian, 2, L, Nkeep, truncationtolerance=0.0)
println("Ground state energy from iterative diagonalization: $(energies[end])")

# part (ii) (2): Obtain the energy expectationvalue <Psi|H|Psi> using the MPO, and compare
# with the energy obtained from iterative diagonalization
function expectationvalue(
    mps::AbstractVector{<:AbstractArray{<:Number, 3}},
    mpo::AbstractVector{<:AbstractArray{<:Number, 4}},
    mps2::AbstractVector{<:AbstractArray{<:Number, 3}}=mps
)
    expval = ones(ComplexF64, 1, 1, 1)
    for (M, W, M2) in zip(mps, mpo, mps2)
        expval = updateLeft(expval, M2, W, M)
    end
    return expval
end

mpoenergy = expectationvalue(mps, mpo)  # Because we have multiple states
groundstateenergy = minimum(real, diag(mpoenergy[:, 1, :]))
println("Ground state energy from MPO: $groundstateenergy")
println("Difference: $(energies[end] - groundstateenergy)")
