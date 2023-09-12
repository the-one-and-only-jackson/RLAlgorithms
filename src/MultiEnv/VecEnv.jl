@with_kw struct VecEnv{E<:AbstractEnv} <: AbstractMultiEnv
    envs::Vector{E}
    dones::Vector{Bool} = fill(false, length(envs))
    truncs::Vector{Bool} = fill(false, length(envs))
    auto_reset::Bool = true
    batch::Bool = false
end

function VecEnv(env_fcn::Function; n_envs=1, kw_args...)
    envs = [env_fcn() for _ in 1:n_envs]
    return VecEnv(; envs, kw_args...)
end

Base.length(e::VecEnv) = length(e.envs)

function CommonRLInterface.reset!(e::VecEnv)
    reset!.(e.envs)
    e.dones .= terminated.(e.envs)
    e.truncs .= truncated.(e.envs)
    nothing
end

function CommonRLInterface.reset!(e::VecEnv, i::Integer)
    env = e.envs[i]
    reset!(env)
    e.dones[i] = terminated(env)
    e.truncs[i] = truncated(env)
    nothing
end

function CommonRLInterface.reset!(e::VecEnv, idxs::AbstractVector{Bool})
    envs = e.envs[idxs]
    reset!.(envs)
    e.dones[idxs] .= terminated.(envs)
    e.truncs[idxs] .= truncated.(envs)
    nothing
end

function CommonRLInterface.act!(e::VecEnv, a)
    r = _act!(e, a)
    e.dones .= terminated.(e.envs)
    e.truncs .= truncated.(e.envs)
    e.auto_reset && reset!.(e.envs[e.dones])
    return r
end

function _act!(e::VecEnv, a::AbstractVector)
    @assert length(e.envs)==length(a) "Number of actions does not match number of envs."
    act!.(e.envs, a)
end

function _act!(e::VecEnv, a::AbstractArray)
    dims = ndims(a)
    @assert length(e.envs)==size(a, dims) "Number of actions does not match number of envs."
    act!.(e.envs, eachslice(a; dims))
end 

CommonRLInterface.terminated(e::VecEnv) = copy(e.dones)
CommonRLExtensions.truncated(e::VecEnv) = copy(e.truncs)

function CommonRLInterface.observe(e::VecEnv; batch=true)
    if batch
        O = observations(e)
        obs = zeros(eltype(O), size(O))
        for (dst,env) in zip(eachslice(obs; dims=ndims(obs)), e.envs)
            dst .= observe(env)
        end
        return obs
    else
        return observe.(e.envs)
    end
end

single_actions(e::VecEnv) = actions(first(e.envs))
single_observations(e::VecEnv) = observations(first(e.envs))

# RL.valid_action_mask(e::VecEnv) = valid_action_mask.(e.envs)
# Note: setup "forward to wrapped" and provided

CommonRLInterface.provided(::typeof(valid_action_mask), e::VecEnv, args...) = provided(valid_action_mask, first(e.envs), args...)
CommonRLInterface.provided(::typeof(valid_actions), e::VecEnv, args...) = provided(valid_actions, first(e.envs), args...)

function CommonRLInterface.valid_action_mask(e::VecEnv; batch=true)
    if batch
        A = actions(e)
        mask = zeros(Bool, size(A))
        for (dst,env) in zip(eachslice(mask; dims=ndims(obs)), e.envs)
            dst .= valid_action_mask(env)
        end
        return obs
    else
        return valid_action_mask.(e.envs)
    end
end

function CommonRLInterface.valid_actions(e::VecEnv; batch=true)
    if batch
        A = actions(e)
        mask = zeros(eltype(A), size(A))
        for (dst,env) in zip(eachslice(mask; dims=ndims(obs)), e.envs)
            dst .= valid_actions(env)
        end
        return obs
    else
        return valid_actions.(e.envs)
    end
end

