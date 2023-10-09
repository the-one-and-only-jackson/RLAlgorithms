abstract type ActorCritic end

function (ac::ActorCritic)(state)
    action, _, _, _ = get_actionvalue(ac, state)
    return action
end

struct DiscreteActorCritic <: ActorCritic
    shared::Chain
    actor::Chain
    critic::Chain
    rng::AbstractRNG
end
Flux.@functor DiscreteActorCritic

struct ContinuousActorCritic{A<:Chain,B<:Chain,C<:Chain,D<:AbstractVector{<:AbstractFloat},E<:AbstractRNG} <: ActorCritic
    shared::A
    actor::B
    critic::C
    log_std::D
    rng::E
    squash::Bool
end
Flux.@functor ContinuousActorCritic

"""
Constructors
"""

function ActorCritic(env; kwargs...)
    if SpaceStyle(actions(env)) == ContinuousSpaceStyle()
        return ContinuousActorCritic(env; kwargs...)
    elseif SpaceStyle(actions(env)) == FiniteSpaceStyle()
        return DiscreteActorCritic(env; kwargs...)
    else
        @assert false "ERROR: Actor crtic construction"
    end
end

function DiscreteActorCritic(env; rng=default_rng(), kwargs...)
    ns = length(single_observations(env))
    na = length(single_actions(env))

    shared, actor, critic = feedforward_feature(ns, na; kwargs...)

    return DiscreteActorCritic(shared, actor, critic, rng)
end

function ContinuousActorCritic(env::AbstractMultiEnv; rng=default_rng(), log_std_init=-0.5, squash=false, kwargs...)
    O = single_observations(env)
    A = single_actions(env)

    @assert A isa Box "This actor-critic model only accepts RLAlgorithms.Spaces.Box observation spaces"
    @assert A isa Box "This actor-critic model only accepts RLAlgorithms.Spaces.Box action spaces"

    @assert length(size(A))==1 "Only vector action spaces supported"

    ns = length(O)
    na = length(A)

    if length(size(O)) == 1
        shared, actor, critic = feedforward_feature(ns, na; kwargs...)
    else
        @assert false "Image observations not yet supported"
    end

    # add code for action scaling

    log_std = fill(log_std_init, na) .|> Float32

    return ContinuousActorCritic(shared, actor, critic, log_std, rng, squash)
end

function feedforward_feature(ns, na;
    shared_dims = [], 
    actor_dims  = [64, 64],
    critic_dims = [64, 64],
    act_fun     = tanh,
    hidden_init = Flux.orthogonal(; gain=sqrt(2)),
    actor_init  = Flux.orthogonal(; gain=0.01),
    critic_init = Flux.orthogonal(; gain=1),
    )

    if isempty(shared_dims)
        actor_dims = [ns; actor_dims; na]
        critic_dims = [ns; critic_dims; 1]
    else
        shared_dims = [ns; shared_dims]
        actor_dims = [shared_dims[end]; actor_dims; na]
        critic_dims = [shared_dims[end]; critic_dims; 1]
    end

    shared = mlp(shared_dims, act_fun, hidden_init)
    actor = mlp(actor_dims, act_fun, hidden_init, actor_init)
    critic = mlp(critic_dims, act_fun, hidden_init, critic_init)

    return shared, actor, critic
end

function mlp(dims, act_fun, hidden_init, head_init = nothing)
    end_idx = isnothing(head_init) ? length(dims) : length(dims)-1
    layers = Dense[Dense(dims[ii] => dims[ii+1], act_fun; init=hidden_init) for ii = 1:end_idx-1]
    if !isnothing(head_init)
        push!(layers, Dense(dims[end-1] => dims[end]; init=head_init))
    end
    return Chain(layers...)
end

"""
get_actionvalue(ac::DiscreteActorCritic, state::AbstractMatrix, action=nothing)
return (action<:AbstractArray, action_log_prob<:AbstractVector, entropy<:AbstractVector, value<:AbstractVector)
"""

function get_feedforward_out(ac::ActorCritic, state::AbstractArray{Float32}) # change state
    shared_out = isempty(ac.shared) ? state : ac.shared(state)
    actor_out = ac.actor(shared_out)
    critic_out = ac.critic(shared_out)
    return actor_out, critic_out
end

function get_actionvalue(
    ac::DiscreteActorCritic, 
    state::AbstractArray{<:Real};
    action::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    action_mask = nothing
    )

    actor_out, value = get_feedforward_out(ac, state)

    if !isnothing(action_mask)
        actor_out += eltype(actor_out)(-1f10) * .!action_mask
    end

    log_probs = actor_out .- logsumexp(actor_out; dims=1)
    probs = softmax(actor_out; dims=1)

    entropy = -sum(log_probs .* probs, dims=1)

    if isnothing(action)
        action = sample_discrete.(ac.rng, eachcol(cpu(probs))) # need to implement for gpu?
    end

    action_log_prob = log_probs[CartesianIndex.(action, 1:length(action))]

    return (action, action_log_prob, entropy, value) .|> vec
end

function get_actionvalue(
    ac::ContinuousActorCritic, 
    state::AbstractArray{<:Real},
    action::Union{Nothing, <:AbstractArray{<:Real}} = nothing;
    log_min = -20,
    log_max = 2,
    kwargs...
    )

    action_mean, value = get_feedforward_out(ac, state)
    log_std = clamp.(ac.log_std, log_min, log_max)
    action_std = exp.(log_std)

    (action, action_log_prob, entropy) = if ac.squash
        get_action_squash(action_mean, log_std, action_std, ac.rng, action)
        # squashed to [-1,1]

    else
        get_action(action_mean, log_std, action_std, ac.rng, action)
    end

    return (action, vec(action_log_prob), vec(entropy), vec(value))
end

function get_action(
    action_mean,
    log_std,
    action_std,
    rng,
    action::Union{Nothing, <:AbstractArray{<:Real}} = nothing
    )

    if isnothing(action)
        stand_normal = randn(rng, eltype(action_mean), size(action_mean))
        action = action_mean .+ action_std .* stand_normal
    else
        stand_normal = (action .- action_mean) ./ action_std
    end

    action_log_prob = normal_logpdf(stand_normal, log_std)

    temp = (1+log(2f0*pi))/2
    entropy = sum(log_std; dims=1) .+ size(log_std,1) * temp

    return (action, action_log_prob, entropy)
end

function get_action_squash(
    action_mean,
    log_std,
    action_std,
    rng,
    action::Union{Nothing, <:AbstractArray{<:Real}} = nothing;
    action_clamp::Float32 = 5f0
    )
    
    if isnothing(action)
        stand_normal = randn(rng, eltype(action_mean), size(action_mean))
        action_normal = action_mean .+ action_std .* stand_normal
        action = tanh.(clamp.(action_normal, -action_clamp, action_clamp)) # clamp for numeric stability
    else
        stand_normal = (inv_tanh(action) .- action_mean) ./ action_std
    end

    action_log_prob = normal_logpdf(stand_normal, log_std) .- sum(log.(1 .- action .^ 2); dims=1)

    entropy = sum(-action_log_prob; dims=1) # estimate

    return (action, action_log_prob, entropy)
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

inv_tanh(x) = log.((1 .+ x)./(1 .- x))/2


