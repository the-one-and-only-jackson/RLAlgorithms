@with_kw struct LoggingWrapper <: AbstractMultiWrapper
    env::AbstractMultiEnv
    current_step::Vector{Int} = zeros(Int, length(env))
    total_step = [0]
    current_reward::Vector{Float32} = zeros(Int, length(env))
    step::Vector{Int} = Int[]
    episode_length::Vector{Int} = Int[]
    reward::Vector{Float32} = Float32[]
end
LoggingWrapper(env) = LoggingWrapper(; env)

Wrappers.wrapped_env(w::LoggingWrapper) = w.env

CommonRLExtensions.info(w::LoggingWrapper) = Dict("steps"=>w.step, "episode_length"=>w.episode_length, "reward"=>w.reward)

function CommonRLInterface.act!(w::LoggingWrapper, a)
    r = act!(w.env, a)

    w.current_reward .+= r

    w.current_step .+= 1
    w.total_step[] += length(w.env)

    dones = terminated(w.env)
    _logging_reset!(w, dones)

    return r
end

function CommonRLInterface.reset!(w::LoggingWrapper, idxs)
    reset!(w.env, idxs)
    _logging_reset!(w, idxs)
    nothing
end

function _logging_reset!(w::LoggingWrapper, idxs)
    n_done = count(idxs)
    iszero(n_done) && return
    append!(w.step, fill(w.total_step[], n_done))
    append!(w.episode_length, w.current_step[idxs])
    append!(w.reward, w.current_reward[idxs])
    w.current_step[idxs] .= 0
    w.current_reward[idxs] .= 0
    return
end

