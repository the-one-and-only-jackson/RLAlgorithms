module Algorithms

using CommonRLInterface
using Flux
using CUDA
using ProgressMeter
using Parameters

using Random: AbstractRNG, default_rng, seed!, randperm
using Statistics: mean, std
using ChainRules: ignore_derivatives, @ignore_derivatives
using LinearAlgebra: norm

using ..Utils
using ..Spaces
using ..MultiEnv
using ..CommonRLExtensions

export 
    ActorCritic,
    get_actionvalue,
    DiscreteActorCritic,
    ContinuousActorCritic 
include("ActorCritics.jl")

include("Buffer.jl")

export 
    solve,
    PPOSolver
include("ppo.jl")

end