@with_kw mutable struct Buffer{
    S<:AbstractArray{<:Real}, 
    A<:AbstractArray{<:Real}, 
    R<:AbstractArray{<:Real}, # size = (n_env, traj_len)
    D<:AbstractArray{Bool}, 
    AM<:Union{Nothing,<:AbstractArray{Bool}}
    }

    s::S
    a::A
    r::R = zeros(Float32, size(s)[end-1:end])
    done::D = zeros(Bool, size(s)[end-1:end])
    trunc::D = zeros(Bool, size(s)[end-1:end])
    a_logprob::R = copy(r)
    action_mask::AM = nothing
    value::R = copy(r)
    next_value::R = copy(r)
    advantages::R = copy(r)
    returns::R = copy(r)
    traj_len::Int
    idx::Int = 1
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

    return Buffer(; s, a, action_mask, traj_len)
end

Flux.@functor Buffer

function sample_batch!(dst::T, src::T, idxs) where {T <: Buffer}
    copy_fun(x, y) = copyto!(x, selectdim(y, ndims(y), idxs))
    fmap(copy_fun, dst, src; exclude=x->x isa AbstractArray)
    nothing
end

function fill_buffer!(env::AbstractMultiEnv, buffer::Buffer, ac::ActorCritic)
    values_t = get_stateactionvalue(env, ac)
    
    for _ in 1:buffer.traj_len
        r = act!(env, values_t.a)
        done = terminated(env)
        trunc = truncated(env)    

        values_t′ = get_stateactionvalue(env, ac)
    
        send_to!(buffer; values_t..., r, done, trunc, next_value=values_t′.value)

        if count(trunc) != 0
            reset!(env, trunc)
            values_t′ = get_stateactionvalue(env, ac)   
        end

        values_t = values_t′
    end    
    nothing
end

function get_stateactionvalue(env, ac)
    s = observe(env)
    action_mask = if provided(valid_action_mask, env)
        valid_action_mask(env)
    else
        nothing
    end
    (a, a_logprob, _, value) = get_actionvalue(ac, s; action_mask)
    return (; s, action_mask, a, a_logprob, value)
end

function send_to!(buffer::Buffer; kwargs...)
    idx = buffer.idx
    for (key, val) in kwargs
        isnothing(val) && continue
        arr = getfield(buffer, key)
        copyto!(selectdim(arr, ndims(arr), idx), val)
    end
    buffer.idx = mod1(1+idx, buffer.traj_len)
    nothing
end

function gae!(buffer::Buffer, γ::Real, λ::Real)
    @unpack r, done, value, next_value, advantages, returns, trunc = buffer

    # Correctly initialize gpu or cpu arr
    next_advantage = similar(advantages, size(advantages,1)) .= 0

    contin = @. !(done || trunc)

    for (advantages, r, contin, value, next_value) in Iterators.reverse(zip(eachcol.((advantages, r, contin, value, next_value))...))
        td = @. r + γ * contin * next_value - value
        next_advantage = @. advantages = td + λ * γ * contin * next_advantage
    end
    @. returns = advantages + value
    nothing
end