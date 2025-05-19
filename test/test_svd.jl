using Test
using LinearAlgebra
import tn_julia
import tn_julia: contract, svdleft, svdright, svdleftright, tensor2MPS
import Random

Random.seed!(1234) # Initialize the random number generator seed for repoducibility

@testset "SVD" begin
    @testset "Obtaining isometries using SVD" begin
        d = 1
        D = 300
        M = rand(D, d, D)
        M4leg = rand(D, d, d, D)

        Lambda, B = svdleft(M)
        @test size(Lambda) == (D, D)
        @test size(B) == (D, d, D)
        @test contract(Lambda, [2], B, [1]) ≈ M
        @test contract(B, [2, 3], conj(B), [2, 3]) ≈ I(D) # B is an isometry

        A, Lambda = svdright(M)
        @test size(A) == (D, d, D)
        @test size(Lambda) == (D, D)
        @test contract(A, [3], Lambda, [1]) ≈ M
        @test contract(A, [1, 2], conj(A), [1, 2]) ≈ I(D) # A is an isometry

        A, Lambda, B = svdleftright(M4leg)
        @test size(A) == (D, d, D)
        @test size(Lambda) == (D, D)
        @test size(B) == (D, d, D)
        @test contract(contract(A, [3], Lambda, [1]), [3], B, [1]) ≈ M4leg
        @test contract(A, [1, 2], conj(A), [1, 2]) ≈ I(D) # A is an isometry
        @test contract(B, [2, 3], conj(B), [2, 3]) ≈ I(D) # B is an isometry
    end

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

        MPS, discardedweight = tensor2MPS(GHZ; Nkeep=2)
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

        MPS, discardedweight = tensor2MPS(GHZ; Nkeep=1)
        @test all(ndims.(MPS) .== 3)
        for M in MPS
            @test size(M) == (1, 2, 1)
        end
        # Now, we actually discarded information.
        @test sum(discardedweight) .≈ 0.5
    end
end
