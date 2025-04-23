import tn_julia: gaussian

@testset "Basic properties of a gaussian" begin
    @test gaussian(0.0) == 1
    @test gaussian(1.0) == exp(-1.0)
    for x in 0:0.1:1
        @test gaussian(x) == gaussian(-x)
    end
end
