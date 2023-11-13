@with_kw struct Buffer{
    S<:AbstractArray{<:Real}, 
    A<:AbstractArray{<:Real}, 
    R<:AbstractArray{<:Real}, # size = (n_env, traj_len)
    D<:AbstractArray{Bool}, 
    AM<:Union{Nothing,<:AbstractArray{Bool}}
    }    

    traj_len::Int
    n_envs::Int
    s::S
    a::A
    r::R = zeros(Float32, 1, n_envs, traj_len)
    done::D = zeros(Bool, 1, n_envs, traj_len)
    trunc::D = copy(done)
    a_logprob::R = copy(r)
    action_mask::AM = nothing
    value::R = copy(r)
    next_value::R = copy(r)
end

function Buffer(env::AbstractMultiEnv, traj_len::Int)
    n_envs = length(env)

    O = single_observations(env)
    s = zeros(eltype(O), size(O)..., n_envs, traj_len)

    A = single_actions(env)
    if SpaceStyle(A) == ContinuousSpaceStyle()
        a = zeros(eltype(A), size(A)..., n_envs, traj_len)
    elseif SpaceStyle(A) == FiniteSpaceStyle()
        a = zeros(eltype(A), ndims(A), n_envs, traj_len)
    else
        @assert "Space Style Error."
    end

    if provided(valid_action_mask, env)
        action_mask = trues(size(A)..., n_envs, traj_len)
        println("Action mask provided.")
    else
        action_mask = nothing
    end

    return Buffer(; s, a, action_mask, traj_len, n_envs)
end

Flux.@functor Buffer

Base.length(b::Buffer) = b.traj_len * b.n_envs

function send_to!(buffer::Buffer, idx; kwargs...)
    for (key, val) in kwargs
        isnothing(val) && continue
        arr = getfield(buffer, key)
        copyto!(selectdim(arr, ndims(arr), idx), val)
    end
    nothing
end
