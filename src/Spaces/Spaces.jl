module Spaces

using StaticArrays, Distributions

export
    FiniteSpaceStyle,
    ContinuousSpaceStyle,
    UnknownSpaceStyle,
    HybridSpaceStyle,
    SpaceStyle,
    NumericArraySpace,
    Box,
    product,
    MultiAgentArraySpace,
    Discrete,
    DistributionSpace,
    TupleSpace,
    wrapped_space


abstract type AbstractSpaceStyle end
struct FiniteSpaceStyle <: AbstractSpaceStyle end
struct ContinuousSpaceStyle <: AbstractSpaceStyle end
struct UnknownSpaceStyle <: AbstractSpaceStyle end
struct HybridSpaceStyle <: AbstractSpaceStyle end
SpaceStyle(::Any) = UnknownSpaceStyle()

const KnownSpaceStyle = Union{FiniteSpaceStyle, ContinuousSpaceStyle, HybridSpaceStyle}

promote_spacestyle(::T, ::T) where {T<:AbstractSpaceStyle} = T()
promote_spacestyle(::FiniteSpaceStyle, ::ContinuousSpaceStyle) = HybridSpaceStyle()
promote_spacestyle(::ContinuousSpaceStyle, ::FiniteSpaceStyle) = HybridSpaceStyle()
promote_spacestyle(::FiniteSpaceStyle, ::HybridSpaceStyle) = HybridSpaceStyle()
promote_spacestyle(::ContinuousSpaceStyle, ::HybridSpaceStyle) = HybridSpaceStyle()
promote_spacestyle(::AbstractSpaceStyle, ::AbstractSpaceStyle) = UnknownSpaceStyle()


abstract type NumericArraySpace end

wrapped_space(space::NumericArraySpace) = space

struct Box{Size,T<:AbstractFloat,N,L} <: NumericArraySpace
    lower::SArray{Size,T,N,L}
    upper::SArray{Size,T,N,L}

    function Box{T}(lower::SArray{Size,TL,N,L}, upper::SArray{Size,TU,N,L}) where {Size,T<:AbstractFloat,N,L,TL,TU}
        _lower = convert(SArray{Size,T,N,L}, lower)
        _upper = convert(SArray{Size,T,N,L}, upper)
        new{Size,T,N,L}(_lower, _upper)
    end

    function Box{T}(lower::AbstractArray, upper::AbstractArray) where {T<:AbstractFloat}
        s = size(lower)
        @assert s == size(upper)
        S = Tuple{s...}
        L = reduce(*, s)
        N = length(s)
        _lower = SArray{S,T,N,L}(lower)
        _upper = SArray{S,T,N,L}(upper)
        new{S,T,N,L}(_lower, _upper)
    end

    Box{T}(; lower::AbstractArray, upper::AbstractArray) where {T} = Box{T}(lower, upper)

    Box(lower::AbstractArray, upper::AbstractArray) = Box{float(promote_type(eltype(lower), eltype(upper)))}(lower, upper)

    Box(; lower::AbstractArray, upper::AbstractArray) = Box(lower, upper)

    Box(T::Type, size::Tuple) = Box{T}(fill(typemin(T), size), fill(typemax(T), size))
    Box(T::Type, size...) = Box(T, size)
end

Base.size(::Box{Size,T,N,L}) where {Size,T,N,L} = StaticArrays.size_to_tuple(Size)
Base.eltype(::Box{Size,T,N,L}) where {Size,T,N,L} = T
Base.ndims(::Box{Size,T,N,L}) where {Size,T,N,L} = N
Base.length(::Box{Size,T,N,L}) where {Size,T,N,L} = L

SpaceStyle(::Box) = ContinuousSpaceStyle()

Base.convert(::Type{Box{T}}, b::Box) where T = Box{T}(b.lower, b.upper)


function product(b1::Box, b2::Box)
    @assert ndims(b1) == ndims(b2)
    @assert size(b1)[1:end-1] == size(b2)[1:end-1]
    lower = cat(b1.lower, b2.lower; dims=ndims(b1))
    upper = cat(b1.upper, b2.upper; dims=ndims(b1)) 
    return Box(lower, upper)
end


struct Discrete{S} <: NumericArraySpace
    base_space::S
end
Discrete(n::T) where {T<:Integer} = Discrete(one(T):n)

SpaceStyle(::Discrete) = FiniteSpaceStyle()
Base.size(::Discrete) = (1,)
Base.eltype(::Discrete{T}) where {T} = eltype(T)
Base.ndims(::Discrete) = 1
Base.length(::Discrete) = 1
Base.collect(d::Discrete) = collect(d.base_space)

wrapped_space(space::Discrete) = space.base_space



struct MultiAgentArraySpace{S<:NumericArraySpace} <: NumericArraySpace
    base_space::S
    n_agents::Int
end

SpaceStyle(space::MultiAgentArraySpace) = SpaceStyle(space.base_space)
Base.size(space::MultiAgentArraySpace) = (size(space.base_space)..., space.n_agents)
Base.eltype(space::MultiAgentArraySpace) = eltype(space.base_space)
Base.ndims(space::MultiAgentArraySpace) = 1 + ndims(space.base_space)
Base.length(space::MultiAgentArraySpace) = space.n_agents * length(space.base_space)
Base.collect(space::MultiAgentArraySpace) = collect(space.base_space)

wrapped_space(space::MultiAgentArraySpace) = space.base_space


struct DistributionSpace{D<:Distribution} <: NumericArraySpace
    base_space::D
end

SpaceStyle(::DistributionSpace{<:Distribution{<:Any, Distributions.Continuous}}) = ContinuousSpaceStyle()
SpaceStyle(::DistributionSpace{<:Distribution{<:Any, Distributions.Discrete}}) = FiniteSpaceStyle()

Base.size(space::DistributionSpace) = size(space.base_space)
Base.eltype(space::DistributionSpace) = eltype(space.base_space)
Base.ndims(space::DistributionSpace) = length(size(space))
Base.length(space::DistributionSpace) = length(space.base_space)
Base.collect(space::DistributionSpace) = collect(space.base_space)

wrapped_space(space::DistributionSpace) = space.base_space


struct TupleSpace{S<:Tuple{Vararg{<:NumericArraySpace}}} <: NumericArraySpace
    base_spaces::S
end
TupleSpace(args...) = TupleSpace(args)

Base.length(space::TupleSpace) = length(space.base_spaces)
wrapped_space(space::TupleSpace) = space.base_spaces
SpaceStyle(space::TupleSpace) = reduce(SpaceStyle, space.base_spaces)

end