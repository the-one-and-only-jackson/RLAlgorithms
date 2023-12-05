# abstract type ActorCritic end

struct ActorCritic{S,A,C}
    shared::S
    actor::A
    critic::C
end
Flux.@functor ActorCritic

struct DiscreteActor{B<:Chain,D<:AbstractRNG}
    actor::B
    rng::D
end
Flux.@functor DiscreteActor

struct ContinuousActor{B<:Chain,D<:AbstractVector{<:AbstractFloat},E<:AbstractRNG}
    actor::B
    log_std::D
    rng::E
    squash::Bool
end
Flux.@functor ContinuousActor

function ActorCritic(env; 
    shared_dims = [], 
    critic_dims = [64, 64],
    act_fun     = tanh,
    hidden_init = Flux.orthogonal(; gain=sqrt(2)),
    critic_init = Flux.orthogonal(; gain=1),
    kwargs...
    )

    A = single_actions(env)
    O = single_observations(env)

    @assert ndims(O) == 1 # only feature vector for now
    ns = length(O)

    shared_out_size = isempty(shared_dims) ? ns : shared_dims[end]

    if !isempty(shared_dims)
        shared_dims = [ns; shared_dims]
    end
    shared = mlp(shared_dims, act_fun, hidden_init)

    actor = Actor(A, shared_out_size; kwargs...)

    critic_dims = [shared_out_size; critic_dims; 1]
    critic = mlp(critic_dims, act_fun, hidden_init, critic_init)

    return ActorCritic(shared, actor, critic) 
end

Actor(args...; kwargs...) = @assert false "ERROR: Actor crtic construction"

function Actor(A::Box, shared_out_size;
    actor_dims  = [64, 64],
    act_fun     = tanh,
    hidden_init = Flux.orthogonal(; gain=sqrt(2)),
    actor_init  = Flux.orthogonal(; gain=0.01),
    log_std_init= -0.5, 
    squash      = false,
    rng         = default_rng(),
    kwargs...
    )
    na = length(A)
    actor_dims = [shared_out_size; actor_dims; na]
    act_net = mlp(actor_dims, act_fun, hidden_init, actor_init)
    log_std = fill(log_std_init, na) .|> Float32
    ContinuousActor(act_net, log_std, rng, squash)
end

function Actor(A::Discrete, shared_out_size;
    actor_dims  = [64, 64],
    act_fun     = tanh,
    hidden_init = Flux.orthogonal(; gain=sqrt(2)),
    actor_init  = Flux.orthogonal(; gain=0.01),
    rng         = default_rng(),
    kwargs...
    )
    actor_dims = [shared_out_size; actor_dims; length(collect(A))]
    act_net = mlp(actor_dims, act_fun, hidden_init, actor_init)
    DiscreteActor(act_net, rng)
end

function Actor(A::TupleSpace, shared_out_size; kwargs...)
    Tuple(Actor(space, shared_out_size; kwargs...) for space in wrapped_space(A))
end

function (ac::ActorCritic)(state)
    action, _, _, _ = get_actionvalue(ac, state)
    return action
end

function mlp(dims, act_fun, hidden_init, head_init = nothing)
    end_idx = isnothing(head_init) ? length(dims) : length(dims)-1
    layers = Dense[Dense(dims[ii] => dims[ii+1], act_fun; init=hidden_init) for ii = 1:end_idx-1]
    if !isnothing(head_init)
        push!(layers, Dense(dims[end-1] => dims[end]; init=head_init))
    end
    return Chain(layers...)
end

function get_actionvalue(
    ac::ActorCritic, 
    state::AbstractArray{<:Real},
    action::Union{Nothing, AbstractArray{<:Real}} = nothing;
    action_mask = nothing
    )

    shared_out = isempty(ac.shared) ? state : ac.shared(state)
    action_info = get_action(ac.actor, shared_out, action; action_mask)
    value = ac.critic(shared_out)

    return (action_info..., value)
end


function get_action(
    ac::DiscreteActor, 
    shared_out::AbstractArray{<:Real},
    action::Union{Nothing, AbstractMatrix{<:Integer}} = nothing;
    action_mask = nothing
    )

    actor_out = ac.actor(shared_out)

    if !isnothing(action_mask)
        actor_out += eltype(actor_out)(-1f10) * .!action_mask
    end

    log_probs = actor_out .- logsumexp(actor_out; dims=1)
    probs = softmax(actor_out; dims=1)

    entropy = -sum(log_probs .* probs, dims=1)

    if isnothing(action)
        action = sample_discrete.(ac.rng, eachcol(cpu(probs))) # need to implement for gpu?
    end

    idxs = CartesianIndex.(action, reshape(1:length(action), size(action)))
    action_log_prob = log_probs[idxs]

    return action, action_log_prob, entropy
end

function get_action(
    ac::ContinuousActor, 
    shared_out::AbstractArray{<:Real},
    action::Union{Nothing, <:AbstractArray{<:Real}} = nothing;
    log_min = -20,
    log_max = 2,
    action_clamp = 5f0,
    kwargs...
    )

    action_mean = ac.actor(shared_out)
    log_std = clamp.(ac.log_std, log_min, log_max)
    action_std = exp.(log_std)

    if ac.squash
        if isnothing(action)
            stand_normal = randn(rng, eltype(action_mean), size(action_mean))
            action = action_mean .+ action_std .* stand_normal
        else
            stand_normal = (action .- action_mean) ./ action_std
        end
    
        action_log_prob = normal_logpdf(stand_normal, log_std)
    
        entropy = sum(log_std; dims=1) .+ size(log_std,1) * (1+log(2f0*pi))/2    
    else
        if isnothing(action)
            stand_normal = randn(rng, eltype(action_mean), size(action_mean))
            action_normal = action_mean .+ action_std .* stand_normal
            action = tanh.(clamp.(action_normal, -action_clamp, action_clamp)) # clamp for numeric stability
        else
            stand_normal = (inv_tanh(action) .- action_mean) ./ action_std
        end
    
        action_log_prob = normal_logpdf(stand_normal, log_std) .- sum(log.(1 .- action .^ 2); dims=1)
    
        entropy = sum(-action_log_prob; dims=1) # estimate    
    end

    return action, action_log_prob, entropy
end

function get_action(ac::Tuple, args...; kwargs...)
    action_info = Tuple(get_action(actor, args...; kwargs...) for actor in ac)

    action          = Tuple(info[1] for info in action_info)
    action_log_prob = Tuple(info[2] for info in action_info)
    entropy         = Tuple(info[3] for info in action_info)

    return action, action_log_prob, entropy
end


"""
Helper functions
"""

function sample_discrete(rng::AbstractRNG, wv::AbstractVector{T}) where T
    t = rand(rng, T) * sum(wv)
    cw = zero(T)
    for (i, w) in enumerate(wv) 
        (cw += w) >= t && return i
    end
    return length(wv)
end
sample_discrete(wv) = sample_discrete(default_rng(), wv)

function normal_logpdf(stand_normal::AbstractArray{T}, log_std::AbstractArray{T}) where T
    num = sum(-stand_normal.^2/2; dims=1)
    den = sum(log_std; dims=1) .+ size(log_std,1) * T(log(2*pi)/2)
    return num .- den
end

inv_tanh(x) = @. log((1 + x)/(1 - x))/2

# Flux helper
(a::Flux.Dense{<:Any,<:CuArray,<:CuArray})(x::Array) = a(cu(x))