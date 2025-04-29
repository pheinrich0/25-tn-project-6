using Test
import tn_julia

@testset "Hamiltonian" begin
    L = 4
    H = tn_julia.tightbindinghamiltonian(L)
    @test size(H) == (L, L)
    @test H' == H

    @test all(diag(H, 0) .== 0)
    @test all(diag(H, 1) .== -1)
    @test all(diag(H, -1) .== -1)

    @test H == -1 .* [
        0 1 0 0;
        1 0 1 0;
        0 1 0 1;
        0 0 1 0
    ]
end
