using Test
using LinearAlgebra
import tn_julia: iterativediagonalization, extendhamiltonian_ff, ff_mpo, mpo_expectation, contract, tightbindinghamiltonian

@testset "ff" begin
    @testset "iterative diagonalization of ff" begin
        L = 2 # chain length
        Nkeep = 4

        extendhamiltonian(H::AbstractMatrix, A::AbstractArray{<:Number, 3}) = 
            extendhamiltonian_ff(H, A; t=1.0)
        energies, _ = iterativediagonalization(extendhamiltonian, 2, L, Nkeep)

        @test energies[end] ≈ -1
    end

    @testset "MPO matches tight-binding matrix at L=2" begin
        L = 2
        d = 2
        W = ff_mpo(L)
        
        ψ = contract(W[1], 3, W[2], 1,)

        #ψ = permutedims(ψ, (2, 5, 3, 6))

        H_mpo = reshape(ψ, d^L, d^L)
        H_mpo_projected = H_mpo[2:3,2:3]
        H_exact = tightbindinghamiltonian(L)
        @test H_mpo_projected ≈ H_exact
    end

    @testset "MPO expectation value matches ground energy at L=100" begin
        L = 100
        Nkeep = 50

        extendhamiltonian(H::AbstractMatrix, A::AbstractArray{<:Number, 3}) =
            extendhamiltonian_ff(H, A; t=1.0)

        energies, MPS = iterativediagonalization(extendhamiltonian, 2, L, Nkeep)
        E_iter = energies[1] # TODO this may be wrong

        MPO = ffchainmpo(L)
        E_mpo = mpo_expectation(MPO, MPS)

        @test isapprox(E_iter, E_mpo; atol=1e-8)
    end
end
