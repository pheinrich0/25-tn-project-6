using Test
import tn_julia: spinlocalspace

@testset "localspace" begin
    @testset "test spin localspace" for spin in [1//2, 1, 3//2]
        commutator(A, B) = A * B - B * A

        Splus, Sminus, Sz, Id = spinlocalspace(spin)

        @test size(Splus) == (2spin + 1, 2spin + 1)
        @test size(Sminus) == (2spin + 1, 2spin + 1)
        @test size(Sz) == (2spin + 1, 2spin + 1)
        @test size(Id) == (2spin + 1, 2spin + 1)

        @test commutator(Splus, Sminus) ≈ Sz
        @test commutator(Sz, Splus) ≈ Splus
        @test commutator(Sz, Sminus) ≈ -Sminus
        @test Id * Sz ≈ Sz
    end
end
