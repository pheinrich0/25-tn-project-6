using Test
using LinearAlgebra
import tn_julia: contract

@testset "SVD" begin
    @testset "Applying SVD to a GHZ3 state" begin
        GHZ3 = zeros(2, 2, 2)
        GHZ3[1, 1, 1] = 1 / sqrt(2)
        GHZ3[2, 2, 2] = 1 / sqrt(2)

        U, S, Vd, discardedweight = tn_julia.svd(GHZ3, [1])
        @test size(U) == (2, 2)
        @test size(S) == (2,)
        @test size(Vd) == (2, 2, 2)
        @test contract(contract(U, [2], Diagonal(S), [1]), [2], Vd, [1]) ≈ GHZ3

        U, S, Vd, discardedweight = tn_julia.svd(GHZ3, [1], Nkeep=1)
        @test size(U) == (2, 1)
        @test size(S) == (1,)
        @test size(Vd) == (1, 2, 2)
        @test discardedweight ≈ 0.5

        U, S, Vd, discardedweight = tn_julia.svd(GHZ3, [1, 2])
        @test size(U) == (2, 2, 2)
        @test size(S) == (2,)
        @test size(Vd) == (2, 2)
        @test contract(contract(U, [3], Diagonal(S), [1]), [3], Vd, [1]) ≈ GHZ3
    end
end
