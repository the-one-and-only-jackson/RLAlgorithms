module ActorCritics

using Flux
using Random: AbstractRNG, default_rng
using CommonRLInterface
using CUDA

export 
    ActorCritic,
    get_actionvalue,
    DiscreteActorCritic,
    ContinuousActorCritic

abstract type ActorCritic end

function (ac::ActorCritic)(state)
    action, _, _, _ = get_actionvalue(ac, state)
    return action
end

const ALLOWED_ARRAY = Union{Array{Float32}, CuArray{Float32}}

struct DiscreteActorCritic <: ActorCritic
    shared::Chain
    actor::Chain
    critic::Chain
    rng::AbstractRNG
end
Flux.@functor DiscreteActorCritic

struct ContinuousActorCritic{T<:ALLOWED_ARRAY} <: ActorCritic
    shared::Chain
    actor::Chain
    critic::Chain
    log_std::T
    rng::AbstractRNG
    squash::Bool
end
Flux.@functor ContinuousActorCritic

"""
Constructors
"""

function DiscreteActorCritic(env; rng=default_rng(), kwargs...)
    if terminated(env) isa Vector
        ns = length(first(observe(env)))
    else
        ns = length(observe(env))
    end
    
    na = length(actions(env))

    shared, actor, critic = feedforward_feature(ns, na; kwargs...)

    return DiscreteActorCritic(shared, actor, critic, rng)
end

function ContinuousActorCritic(env; rng=default_rng(), log_std_init=-0.5, squash=false, kwargs...)
    if terminated(env) isa Vector
        ns = length(first(observe(env)))
    else
        ns = length(observe(env))
    end
    
    if actions(env) isa Vector{<:Vector}
        na = length(actions(env))
    else
        na = 1
    end

    shared, actor, critic = feedforward_feature(ns, na; kwargs...)

    log_std = fill(log_std_init, na) .|> Float32

    return ContinuousActorCritic(shared, actor, critic, log_std, rng, squash)
end

function mlp(dims, act_fun, hidden_init, head_init = nothing)
    end_idx = isnothing(head_init) ? length(dims) : length(dims)-1
    layers = Dense[Dense(dims[ii] => dims[ii+1], act_fun; init=hidden_init) for ii = 1:end_idx-1]
    if !isnothing(head_init)
        push!(layers, Dense(dims[end-1] => dims[end]; init=head_init))
    end
    return Chain(layers)
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

"""
get_actionvalue(ac::DiscreteActorCritic, state::AbstractMatrix, action=nothing)
return (action<:AbstractArray, action_log_prob<:AbstractVector, entropy<:AbstractVector, value<:AbstractVector)
"""

function get_feedforward_out(ac::ActorCritic, state::T)::Tuple{T,T} where {T<:ALLOWED_ARRAY}
    shared_out = isempty(ac.shared) ? state : ac.shared(state)
    actor_out = ac.actor(shared_out)
    critic_out = ac.critic(shared_out)
    return actor_out, critic_out
end

function get_actionvalue(ac::DiscreteActorCritic, state::ALLOWED_ARRAY, action_idx::Union{Nothing, AbstractVector{<:Integer}}=nothing)
    actor_out, value = get_feedforward_out(ac, state)

    log_probs = actor_out .- logsumexp(actor_out; dims=1)
    probs = softmax(log_probs; dims=1)
    entropy = -sum(log_probs .* probs, dims=1)

    if isnothing(action_idx)
        action_idx = sample_discrete.(eachcol(cpu(probs))) # need to implement for gpu?
    end

    action_log_prob = log_probs[CartesianIndex.(action_idx, 1:length(action_idx))]

    return (action_idx, action_log_prob, entropy, value) .|> vec
end
# function get_actionvalue(ac::DiscreteActorCritic, state::AbstractVector, action=nothing; args...)
#     mat_results = get_actionvalue(ac, reshape(state, :, 1), action; args...)
#     return first.(mat_results)
# end

function get_actionvalue(ac::ContinuousActorCritic, state::ALLOWED_ARRAY, action::Union{Nothing, ALLOWED_ARRAY}=nothing; action_clamp=tanh(3f0))
    action_mean, value = get_feedforward_out(ac, state)
    log_std = clamp.(ac.log_std, -20, 2)
    action_std = exp.(log_std)

    if isnothing(action)
        stand_normal = randn(ac.rng, eltype(action_mean), size(action_mean))
        action = action_mean .+ action_std .* stand_normal
        if ac.squash
            action = clamp.(tanh.(action), -action_clamp, action_clamp) # clamp for numeric stability
        end
    else
        action_normal = ac.squash ? inv_tanh(action) : action
        stand_normal = (action_normal .- action_mean) ./ action_std
    end

    action_log_prob = normal_logpdf(stand_normal, log_std)
    if ac.squash
        action_log_prob = action_log_prob .- sum(log.(1 .- action .^ 2); dims=1)
        entropy = sum(-action_log_prob; dims=1) # estimate
    else
        entropy = sum(log_std; dims=1) .+ size(log_std,1)*(1+log(2f0*pi))/2 # exact
    end

    return (action, vec(action_log_prob), vec(entropy), vec(value))
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


end