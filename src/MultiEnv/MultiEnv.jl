module MultiEnv

using CommonRLInterface
using CommonRLInterface.Wrappers
using Statistics
using Parameters: @with_kw
using ..Spaces
using ..CommonRLExtensions

export
    AbstractMultiEnv,
    single_actions,
    single_observations

abstract type AbstractMultiEnv <: AbstractEnv end

function single_actions end
function single_observations end
# Base.length

CommonRLInterface.actions(e::AbstractMultiEnv) = MultiAgentArraySpace(single_actions(e), length(e))
CommonRLInterface.observations(e::AbstractMultiEnv) = MultiAgentArraySpace(single_observations(e), length(e))

export VecEnv
include("VecEnv.jl")

export AbstractMultiWrapper
include("Wrappers/AbstractMultiWrapper.jl")

export ObsNorm, RewNorm
include("Wrappers/norm.jl")

export LoggingWrapper
include("Wrappers/logging.jl")

include("CommonRLExtensions.jl")

end