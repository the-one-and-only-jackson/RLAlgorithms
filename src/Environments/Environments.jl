module Environments

using CommonRLInterface
using Parameters: @with_kw
using StaticArrays: SA, MVector, @MVector
using Distributions

using ..Spaces
using ..CommonRLExtensions

include("mountaincar.jl")
export MountainCar

include("pendulum.jl")
export Pendulum


end