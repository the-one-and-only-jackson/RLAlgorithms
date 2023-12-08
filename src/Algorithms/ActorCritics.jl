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

struct ContinuousActor{T<:AbstractFloat, B<:Chain,D<:AbstractVector{T},E<:AbstractRNG}
    actor::B
    log_std::D
    rng::E
    squash::Bool
    log_min::T
    log_max::T
    action_clamp::T 
end
Flux.@functor ContinuousActor

struct TupleActor{S<:Chain, A<:Tuple}
    shared::S
    actors::A
end
Flux.@functor TupleActor

"""
ActorCritic

kwargs:
    shared_dims
    critic_dims
    actor_dims
    shared_actor_dims
    actor_init
    critic_init
    rng
    log_std_init
    squash
    
"""
function ActorCritic(env; kwargs...)
    shared, shared_out_size = SharedNet(single_observations(env))
    actor = Actor(single_actions(env), shared_out_size; kwargs...)
    critic = Critic(shared_out_size; kwargs...)
    return ActorCritic(shared, actor, critic) 
end

function SharedNet(O::Box; shared_dims = [], kwargs...)
    if ndims(O) == 1 # only feature vector for now
        ns = length(O)

        if !isempty(shared_dims)
            shared_dims = [ns; shared_dims]
        end
        shared = mlp(shared_dims)

        shared_out_size = isempty(shared_dims) ? ns : shared_dims[end]
    elseif ndims(O) == 2
        @assert false "2 dimensions in size(single_observations(env)) not supported.
        If you are using image data, dimensions must be (H,W,C)"
    elseif ndims(O) == 3 # Image data
        @assert false "Image data not yet supported automatically"
        # DQN: Input (84, 84, 4, N)
        Chain(
            Conv((8,8), 4=>32, relu; stride=4),
            Conv((4,4), 32=>64, relu; stride=2),
            Conv((3,3), 64=>64, relu; stride=1),
            Flux.flatten
        )
    else
        @assert false "More than 3 dimensions in size(single_observations(env))"
    end

    return shared, shared_out_size
end

function Critic(input_size;
    critic_dims = [64, 64],
    critic_init = Flux.orthogonal(; gain=1),
    kwargs...
    )

    critic_dims = [input_size; critic_dims; 1]
    mlp(critic_dims; head_init=critic_init)
end

Actor(args...; kwargs...) = @assert false "ERROR: Actor crtic construction"

function Actor(A::Box, input_size;
    actor_dims   = [64, 64],
    actor_init   = Flux.orthogonal(; gain=0.01),
    rng          = default_rng(),
    log_std_init = -0.5, 
    squash       = false,
    log_min = -20,
    log_max = 2,
    action_clamp = 5f0,
    kwargs...
    )
    na = length(A)
    act_net = mlp([input_size; actor_dims; na]; head_init=actor_init, kwargs...)
    log_std = fill(Float32(log_std_init), na)
    ContinuousActor(act_net, log_std, rng, squash, log_min, log_max, action_clamp)
end

function Actor(A::Discrete, input_size;
    actor_dims = [64, 64],
    actor_init = Flux.orthogonal(; gain=0.01),
    rng        = default_rng(),
    kwargs...
    )
    na = length(collect(A))
    act_net = mlp([input_size; actor_dims; na]; head_init=actor_init, kwargs...)
    DiscreteActor(act_net, rng)
end

function Actor(A::TupleSpace, input_size; shared_actor_dims, kwargs...)
    if !isempty(shared_actor_dims)
        shared_actor_dims = [input_size; shared_actor_dims]
        shared_out_size = shared_actor_dims[end]
    else
        shared_out_size = input_size
    end
    shared = mlp(shared_actor_dims; kwargs...)
    actors = Tuple(Actor(space, shared_out_size; kwargs...) for space in wrapped_space(A))
    TupleActor(shared, actors)
end

function (ac::ActorCritic)(state)
    action, _, _, _ = get_actionvalue(ac, state)
    return action
end

function mlp(
    dims; 
    act_fun = tanh, 
    hidden_init = Flux.orthogonal(; gain=sqrt(2)),
    head_init = nothing, 
    kwargs...)

    end_idx = isnothing(head_init) ? length(dims) : length(dims)-1
    layers = Dense[Dense(dims[ii] => dims[ii+1], act_fun; init=hidden_init) for ii = 1:end_idx-1]
    if !isnothing(head_init)
        push!(layers, Dense(dims[end-1] => dims[end]; init=head_init))
    end
    return Chain(layers...)
end

function get_actionvalue(ac::ActorCritic, state, action = nothing; action_mask = nothing)
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
    )

    action_mean = ac.actor(shared_out)
    log_std = clamp.(ac.log_std, ac.log_min, ac.log_max)
    action_std = exp.(log_std)

    if isnothing(action)
        stand_normal = randn(ac.rng, eltype(action_mean), size(action_mean))
        action_normal = action_mean .+ action_std .* stand_normal

        action = if ac.squash
            tanh.(clamp.(action_normal, -ac.action_clamp, ac.action_clamp))
        else
            action_normal
        end
    else
        action_normal = if ac.squash
            inv_tanh(action)
        else
            action
        end

        action_normal = ac.squash ? inv_tanh(action) : action
        stand_normal = (action_normal .- action_mean) ./ action_std
    end

    action_log_prob = if ac.squash
        action_log_prob = normal_logpdf(stand_normal, log_std) .- sum(log.(1 .- action .^ 2); dims=1)
    else
        normal_logpdf(stand_normal, log_std)
    end

    entropy = if ac.squash
        sum(-action_log_prob; dims=1) # estimate  
    else
        sum(log_std; dims=1) .+ size(log_std,1) * (1+log(2f0*pi))/2    
    end

    return action, action_log_prob, entropy
end

function get_action(actor::TupleActor, input, actions; kwargs...)
    shared_out = isempty(actor.shared) ? input : actor.shared(input)
    if isnothing(actions)
        actions = Tuple(nothing for _ in 1:length(actor.actors))
    end
    action_info = Tuple(get_action(actors, shared_out, action; kwargs...) for (actors,action) in zip(actor.actors,actions))
    action, action_log_prob, entropy = Tuple(Tuple(info[i] for info in action_info) for i in 1:3)
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