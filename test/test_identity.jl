using Test
using LinearAlgebra
import tn_julia: identity, contract
import Random

# Set the random seed for reproducibility
Random.seed!(1234)


@testset "identity" begin
    @testset "2-leg identity" begin
        A = rand(3, 4, 5)

        for leg in ndims(A)
            @test contract(A, [leg], identity(A, leg), [1]) == A
        end
    end

    @testset "3-leg identity" begin
        A = rand(3, 4, 5)
        C = I(2)
        B = identity(A, 3, C, 2)

        ABC = contract(
            A, [3],
            contract(B, [2], C, [1]), [1]
        )

        @test size(ABC) == (3, 4, 10, 2)

        # ABC should now contain two copies of A.
        @test ABC[:, :, 1:5, 1] ≈ A
        @test ABC[:, :, 6:10, 2] ≈ A
    end
end
