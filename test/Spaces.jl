@testset "Spaces" begin
    using RLAlgorithms.Spaces

    @testset "Box" begin
        b = Box([1,2], [3,4])
        @test eltype(b) == Float64
        @test size(b) == (2,)
        @test SpaceStyle(b) == ContinuousSpaceStyle()

        @test eltype(Box([1f0, 2f0], [3f0, 4f0])) == Float32
        @test eltype(Box{Float32}([1, 2], [3, 4])) == Float32
        @test eltype(Box{Float32}([1., 2.], [3., 4.])) == Float32

        @test size(Box([1 2; 3 4], [5 6; 7 8])) == (2,2)

        @test Box([1, 2], [3, 4]) == Box(lower=[1, 2], upper=[3, 4])
        @test Box{Float32}([1, 2], [3, 4]) == Box{Float32}(lower=[1, 2], upper=[3, 4])

        @test convert(Box{Float32}, Box([1,2], [3,4])) == Box([1f0, 2f0], [3f0, 4f0])

        b1 = Box([1, 2], [3, 4])
        b2 = Box([5, 6], [7, 8])
        @test product(b1, b2) == Box([1, 2, 5, 6], [3, 4, 7, 8])
    end

    

end