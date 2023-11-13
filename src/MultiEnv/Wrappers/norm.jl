@with_kw mutable struct RunningStats{T<:AbstractArray{<:AbstractFloat}}
    k::Int = 0
    M::T = zeros(T, 1)
    S::T = zeros(T, 1)
end
RunningStats(T, sz) = RunningStats(0, zeros(T, sz), zeros(T, sz))
function (rs::RunningStats)(x)
    rs.k += 1
    if isone(rs.k)
        rs.M .= x
    else
        x_diff = x .- rs.M
        rs.M .+= x_diff / rs.k
        rs.S .+= x_diff .* (x .- rs.M)
    end
    nothing
end
Statistics.mean(rs::RunningStats) = copy(rs.M)
Statistics.var(rs::RunningStats) = (rs.k==1) ? fill!(similar(rs.S), 0) : rs.S/(rs.k - 1)
Statistics.std(rs::RunningStats) = sqrt.(var(rs)) # bad if variance is zero

@with_kw struct ObsNorm{E<:AbstractMultiEnv, RS<:RunningStats} <: AbstractMultiWrapper
    env::E
    obs_stats::RS = RunningStats(eltype(single_observations(env)), size(single_observations(env)))
    s_lim::Float32 = 10f0
    @assert s_lim > 0
end
Wrappers.wrapped_env(wrap::ObsNorm) = wrap.env

CommonRLExtensions.info(wrap::ObsNorm) = Dict(
    "mean"=>mean(wrap.obs_stats), 
    "std"=>std(wrap.obs_stats), 
)

function CommonRLInterface.observe(wrap::ObsNorm)
    s_in = observe(wrap.env)
    _wrap_obs!(wrap.obs_stats, s_in, wrap.s_lim)
    return s_in
end

function _wrap_obs!(obs_stats, s_in::AbstractArray{<:Real}, s_lim)
    itr = eachslice(s_in; dims=ndims(s_in))
    _wrap_obs!(obs_stats, itr, s_lim)
end
function _wrap_obs!(obs_stats, s_in, s_lim; eps=1f-8)
    for s in s_in
        obs_stats(s)
    end
    for s in s_in
        s .-= mean(obs_stats)
        s ./= sqrt.(var(obs_stats) .+ eps)
        clamp!(s, -s_lim, s_lim)
    end
    nothing
end


@with_kw struct RewNorm{E<:AbstractMultiEnv, RS<:RunningStats} <: AbstractMultiWrapper
    env::E
    rew_stats::RS = RunningStats(Float32, 1)
    returns::Vector{Float32} = zeros(Float32, length(env))
    discount::Float32 = 1
    epsilon::Float32 = 1f-8
end
Wrappers.wrapped_env(e::RewNorm) = e.env

function CommonRLInterface.act!(wrap::RewNorm, a)
    r = act!(wrap.env, a)
    wrap.returns .*= wrap.discount .* .!(terminated(wrap.env) .|| truncated(wrap.env))
    wrap.returns .+= r 
    wrap.rew_stats.(wrap.returns)
    return r ./ sqrt.(var(wrap.rew_stats) .+ wrap.epsilon)
end