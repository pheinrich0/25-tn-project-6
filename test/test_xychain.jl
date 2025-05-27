using Test
using LinearAlgebra
import tn_julia: spinlocalspace, xypairhamiltonian, extendhamiltonian

@testset "XY chain hamiltonian" begin
    @testset "XY pair hamiltonian" begin
        H = xypairhamiltonian()
        @test size(H) == (4, 4)
        @test all(eigvals((H + H')/2) ≈ [-0.5, 0.0, 0.0, 0.5])
    end

    @testset "XY pair hamiltonian, generated from extendhamiltonian" begin
        Splus, Sminus, Sz, Id = spinlocalspace(1 // 2)
        H1 = zeros(size(Id))
        A1 = tn_julia.identity(ones(1, 1), 2, H1, 1)
        H2, A2 = extendhamiltonian(H1, A1)

        H = xypairhamiltonian()
        @test H2 ≈ H
    end
end
