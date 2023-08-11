module Spaces

using StaticArrays, Distributions

export
    FiniteSpaceStyle,
    ContinuousSpaceStyle,
    UnknownSpaceStyle,
    SpaceStyle,
    NumericArraySpace,
    Box,
    product,
    MultiAgentArraySpace,
    Discrete,
    DistributionSpace


abstract type AbstractSpaceStyle end
struct FiniteSpaceStyle <: AbstractSpaceStyle end
struct ContinuousSpaceStyle <: AbstractSpaceStyle end
struct UnknownSpaceStyle <: AbstractSpaceStyle end
SpaceStyle(::Any) = UnknownSpaceStyle()


abstract type NumericArraySpace{T<:Number} end
Base.eltype(::NumericArraySpace{T}) where T = T

Base.length(s::NumericArraySpace) = prod(size(s))

struct Box{T} <: NumericArraySpace{T}
    lower::SArray
    upper::SArray
    Box{T}(lower::U, upper::U) where {T<:Real, U<:StaticArray{<:Tuple,T}} = new(lower, upper)
end

function Box(lower::AbstractArray{T1}, upper::AbstractArray{T2}) where {T1<:Real, T2<:Real}
    @assert size(lower) == size(upper)
    S = Tuple{size(lower)...}
    T = promote_type(T1, T2) |> float
    _lower = SArray{S, T}(lower)
    _upper = SArray{S, T}(upper)
    return Box{T}(_lower, _upper)
end

Base.size(b::Box) = size(b.lower)

SpaceStyle(::Box) = ContinuousSpaceStyle()

function product(b1::Box, b2::Box)
    dims = length(size(b1))
    @assert dims == length(size(b2))
    lower = cat(b1.lower, b2.lower; dims)
    upper = cat(b1.upper, b2.upper; dims)
    return Box(lower, upper)
end


struct Discrete{T} <: NumericArraySpace{T}
    base_space
end
function Discrete(space::AbstractArray{T}) where {T<:Number}
    return Discrete{T}(space)
end
function Discrete(n::T) where {T<:Integer}
    return Discrete{T}(one(T):n)
end

SpaceStyle(::Discrete) = FiniteSpaceStyle()
Base.size(space::Discrete) = (length(space.base_space),)
Base.length(space::Discrete) = length(space.base_space)
Base.ndims(::Discrete) = 1


struct MultiAgentArraySpace{T} <: NumericArraySpace{T}
    base_space::NumericArraySpace
    n_agents
end
function MultiAgentArraySpace(base_space::NumericArraySpace{T}, n_agents) where {T}
    return MultiAgentArraySpace{T}(base_space, n_agents)
end

SpaceStyle(space::MultiAgentArraySpace) = SpaceStyle(space.base_space)
Base.size(space::MultiAgentArraySpace) = (size(space.base_space)..., space.n_agents)


struct DistributionSpace{T} <: NumericArraySpace{T}
    d::Distribution
end
function DistributionSpace(d::Distribution)
    return DistributionSpace{eltype(d)}(d)
end
Base.size(d::DistributionSpace) = (length(d.d),)
SpaceStyle(::DistributionSpace) = ContinuousSpaceStyle() # BAD! Wrong! Fix later!


end