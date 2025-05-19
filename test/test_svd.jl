using Test
using LinearAlgebra
import tn_julia: contract

@testset "SVD" begin
    @testset "SVD of a GHZ3 state" begin
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

    @testset "Generating an MPS using SVD" begin
        GHZ = zeros(fill(2, 10)...)
        GHZ[begin] = 1 / sqrt(2)
        GHZ[end] = 1 / sqrt(2)

        MPS, discardedweight = tn_julia.tensor2MPS(GHZ)
        # Since we didn't truncate, the bond dimensions grow exponentially towards the
        # middle.
        @test all(ndims.(MPS) .== 3)
        for (bond, M) in enumerate(MPS)
            Dleft = min(2^(bond - 1), 2^(10 - bond + 1))
            Dright = min(2^(bond), 2^(10 - bond))
            @test size(MPS[bond]) == (Dleft, 2, Dright)
        end
        @test all(discardedweight .≈ 0.0)

        MPS, discardedweight = tn_julia.tensor2MPS(GHZ; Nkeep=2)
        @test all(ndims.(MPS) .== 3)
        # Since Nkeep = 2, the bond dimensions are now limited to 2.
        @test size(MPS[1]) == (1, 2, 2)
        for M in MPS[2:end-1]
            @test size(M) == (2, 2, 2)
        end
        @test size(MPS[end]) == (2, 2, 1)
        # Because there is an exact representation of GHZ as an MPS with D=2, the discarded
        # weight should still be 0.
        @test all(discardedweight .≈ 0.0)

        MPS, discardedweight = tn_julia.tensor2MPS(GHZ; Nkeep=1)
        @test all(ndims.(MPS) .== 3)
        for M in MPS
            @test size(M) == (1, 2, 1)
        end
        # Now, we actually discarded information.
        @test sum(discardedweight) .≈ 0.5
    end
end
