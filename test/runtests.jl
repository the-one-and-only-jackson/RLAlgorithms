using RLAlgorithms
using Test

@testset "Spaces" begin
    using RLAlgorithms.Spaces

    @testset "Box" begin
        for T in [Int32, Int64, Float32, Float64]
            b = Box(T[1,2,3], T[4,5,6])
            @test eltype(b) <: float(T)
            @test size(b) == (3,)
            @test SpaceStyle(b) == ContinuousSpaceStyle()
        end

        b1 = Box([1,2],[3,4])
        b2 = Box([5,6],[7,8])
        @test product(b1,b2) == Box([1,2,5,6],[3,4,7,8])
    end

    @testset "Discrete" begin
        for T in [Int32, Int64, Float32, Float64]
            d = Discrete(T[4,5,6])
            @test eltype(d) <: T
            @test size(d) == (1,)
            @test SpaceStyle(d) == FiniteSpaceStyle()
            @test length(d) == 3
        end

        d = Discrete(4)
        @test length(d) == 4
    end

    @testset "Multi Agent" begin
        b = Box([1,2,3], [4,5,6])
        s = MultiAgentArraySpace(b, 4)
        @test SpaceStyle(s) == ContinuousSpaceStyle()
        @test eltype(s) == eltype(b)
        @test size(s) == (3,4)

        d = Discrete([4,5,6])
        s = MultiAgentArraySpace(d, 4)
        @test SpaceStyle(s) == FiniteSpaceStyle()
        @test eltype(s) == eltype(d)
        @test size(s) == (1,4)
    end
end