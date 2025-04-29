using Test
import tn_julia: updateLeft

@testset "updateLeft dimensions" begin
    D = 4
    w = 2
    d = 3

    A = ones(D, d, D)
    C = ones(D, w, D)
    X = ones(w, d, w, d)

    Cnew = updateLeft(C, A, X, A)
    @test size(Cnew) == (D, w, D)
    @test all(Cnew .== D^2 * w * d^2)
end

@testset "updateLeft applied to isometry" begin
    # This is a randomly generated isometry.
    U = [
        -0.254305-0.232958im   0.157915+0.161259im  -0.0210839-0.373337im       0.18934+0.238625im    0.0330388+0.563119im    -0.254522-0.121703im    0.166111-0.174974im  0.0454926-0.37341im
        -0.167682-0.260876im  -0.217791+0.452701im  0.00655978+0.281734im     -0.101294-0.189124im    -0.122316+0.288483im   -0.0162223+0.20048im     0.141277-0.190769im  -0.415862+0.397589im
        -0.173306-0.234211im  -0.365222+0.155755im   -0.169878-0.184363im      0.338541-0.0173559im  -0.0844374-0.0694863im   -0.130047+0.0175316im  -0.652434+0.115357im   0.273858+0.191185im
        -0.286198-0.362636im   0.161106-0.125207im  -0.0383339-0.0433097im  -0.00618485-0.238045im     -0.22777-0.35831im      0.120278+0.464906im     0.16506-0.363186im   0.280108-0.190855im
    ]

    A = reshape(U', (4, 2, 4))
    C = reshape(I(4), (4, 1, 4))
    X = reshape(I(2), (1, 2, 1, 2))
    @test updateLeft(C, A, X, A) ≈ reshape(I(4), (2, 1, 2))
end
