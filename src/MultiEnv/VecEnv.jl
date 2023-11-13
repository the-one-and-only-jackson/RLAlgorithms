@with_kw struct VecEnv{E<:AbstractEnv} <: AbstractMultiEnv
    envs::Vector{E}
    batch::Bool = true
end

function VecEnv(env_fcn::Function; n_envs=1, kw_args...)
    envs = [env_fcn() for _ in 1:n_envs]
    return VecEnv(; envs, kw_args...)
end

Base.length(e::VecEnv) = length(e.envs)

CommonRLInterface.reset!(e::VecEnv) = reset!(e::VecEnv, :)
function CommonRLInterface.reset!(e::VecEnv, idxs)
    reset!.(e.envs[idxs])
    nothing
end

function CommonRLInterface.act!(e::VecEnv, a::AbstractArray)
    @assert length(e.envs)==size(a, ndims(a)) "Number of actions does not match number of envs."
    # r = act!.(e.envs, eachslice(a; dims=ndims(a)))
    r = zeros(Float32, length(e.envs))
    Threads.@threads for i in 1:length(e.envs)
        r[i] = act!(e.envs[i], selectdim(a, ndims(a), i))
    end 
    return r
end

CommonRLInterface.terminated(e::VecEnv) = terminated.(e.envs)
CommonRLExtensions.truncated(e::VecEnv) = truncated.(e.envs)

function CommonRLInterface.observe(e::VecEnv; batch=e.batch)
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

function CommonRLInterface.valid_action_mask(e::VecEnv; batch=e.batch)
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

function CommonRLInterface.valid_actions(e::VecEnv; batch=e.batch)
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

