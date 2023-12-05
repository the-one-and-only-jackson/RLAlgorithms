@with_kw struct Buffer{
    S<:AbstractArray{<:Real}, 
    A<:Union{AbstractArray{<:Real}, Tuple{Vararg{<:AbstractArray{<:Real}}}}, 
    R<:AbstractArray{<:Real}, # size = (n_env, traj_len)
    D<:AbstractArray{Bool}, 
    AP<:Union{AbstractArray{<:Real}, Tuple{Vararg{<:AbstractArray{<:Real}}}},
    AM<:Union{Nothing,<:AbstractArray{Bool}, Tuple{Vararg{<:AbstractArray{<:Real}}}}
    }    

    traj_len::Int
    n_envs::Int
    s::S
    a::A
    r::R = zeros(Float32, 1, n_envs, traj_len)
    done::D = zeros(Bool, 1, n_envs, traj_len)
    trunc::D = copy(done)
    a_logprob::AP = A isa Tuple ? Tuple(copy(r) for _ in eachindex(A)) : copy(r)
    action_mask::AM = nothing
    value::R = copy(r)
    next_value::R = copy(r)
end

function Buffer(env::AbstractMultiEnv, traj_len::Int)
    n_envs = length(env)

    O = single_observations(env)
    s = zeros(eltype(O), size(O)..., n_envs, traj_len)

    A = single_actions(env)
    action_mask = nothing

    if A isa Box
        a = zeros(eltype(A), size(A)..., n_envs, traj_len)
    elseif A isa Discrete
        a = zeros(eltype(A), size(A)..., n_envs, traj_len)
        if provided(valid_action_mask, env)
            action_mask = trues(length(collect(A)), n_envs, traj_len)
        end
    elseif A isa TupleSpace
        @assert !provided(valid_action_mask, env) "Need to incorporate action masking"
        a = Tuple(zeros(eltype(space), size(space)..., n_envs, traj_len) for space in wrapped_space(A))
    else
        @assert "Space Style Error."
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
