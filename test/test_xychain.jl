using Test
using LinearAlgebra
import tn_julia

@testset "XY pair hamiltonian" begin
    H = tn_julia.xypairhamiltonian()
    @test size(H) == (4, 4)
    @test all(eigvals((H + H')/2) ≈ [-0.5, 0.0, 0.0, 0.5])
end
