using Test
using LinearAlgebra
import tn_julia: tensor2MPS, leftcanonical!, rightcanonical!, sitecanonical!

@testset "MPS" begin
    @testset "Generating an MPS from a GHZ state" begin
        GHZ = zeros(fill(2, 10)...)
        GHZ[begin] = 1 / sqrt(2)
        GHZ[end] = 1 / sqrt(2)

        MPS, discardedweight = tensor2MPS(GHZ)
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

    @testset "Bring the tensor into various canonical forms" begin
        d = 2
        D = 300
        L = 100

        MPS = [
            rand(1, d, D),
            [rand(D, d, D) for _ in 1:L-2]...,
            rand(D, d, 1)
        ]

        @testset "right canonical form" begin
            rightcanonical!(MPS)
            for M in MPS[2:end]
                @test contract(M, [2, 3], M, [2, 3]) ≈ I(size(M, 1)) # M is a right isometry
            end
        end

        # Normalize MPS:
        Lambda, B = svdright(MPS[1]) # Lambda is a 1x1 matrix containing the norm of MPS[1]
        MPS[1] = B

        @testset "left canonical form" begin
            leftcanonical!(MPS)
            for M in MPS[1:end]
                @test contract(M, [1, 2], M, [1, 2]) ≈ I(size(M, 3)) # M is a left isometry
            end
        end

        rightcanonical!(MPS)

        @testset "site canonical form" begin
            center = div(L, 2)
            sitecanonical!(MPS, center)
            for M in MPS[1:center-1]
                @test contract(M, [1, 2], M, [1, 2]) ≈ I(size(M, 3)) # M is a left isometry
            end
            for M in MPS[center+1:end]
                @test contract(M, [2, 3], M, [2, 3]) ≈ I(size(M, 1)) # M is a right isometry
            end
        end
    end
end
