import tn_julia: contract

@testset "Basic properties of the contract function" begin
    A = [0.917561 0.198191; 0.892127 0.616371]
    B = [0.0598884 0.722894; 0.927395 0.533697]
    Id = [1 0; 0 1]
    @test contract(A, 2, B, 1) == A * B
    @test contract(B, 1, A, 2, (2, 1)) == A * B
    @test contract(A, 2, Id, 1) == A
end