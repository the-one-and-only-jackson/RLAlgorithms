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

struct SDEActor{T<:AbstractFloat, A<:Chain,B<:Chain,C<:Chain,E<:AbstractRNG}
    shared_net::A
    actor_net::B
    log_std_net::C
    rng::E
    squash::Bool
    log_min::T
    log_max::T
    action_clamp::T 
end
Flux.@functor SDEActor

struct TupleActor{S<:Chain, A<:Tuple}
    shared::S
    actors::A
end
Flux.@functor TupleActor

struct Critic{NET<:Chain, L, F, G}
    net::NET
    loss::L
    critic_loss_transform::F
    inv_critic_loss_transform::G
end
Flux.@functor Critic


"""
ActorCritic

kwargs:
    shared
    shared_dims
    critic_dims
    actor_dims
    shared_actor_dims
    actor_init
    critic_init
    rng
    log_std_init
    squash
    critic_type
    categorical_values
    critic_loss_transform
    inv_critic_loss_transform
    sde
"""
function ActorCritic(env; shared=nothing, kwargs...)
    if isnothing(shared)
        shared, shared_out_size = SharedNet(single_observations(env))
    else
        O = single_observations(env)
        if O isa Tuple
            sz = (size(O)..., 1)
        else
            sz = map(space->(size(space)..., 1), wrapped_space(O))
        end
        out_size = Flux.outputsize(shared, sz)
        @assert length(out_size) == 2 "Shared layer should output size (X by N) where N is the batch size"
        shared_out_size = out_size[1]
    end

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

Actor(args...; kwargs...) = @assert false "ERROR: Actor crtic construction"

function Actor(A::Box, input_size;
    actor_dims   = [64, 64],
    actor_init   = Flux.orthogonal(; gain=0.01),
    rng          = default_rng(),
    log_std_init = -0.5, 
    squash       = false,
    log_min = -20f0,
    log_max = 2f0,
    action_clamp = 5f0,
    sde = false,
    kwargs...
    )
    na = length(A)
    if sde
        act_net = mlp([input_size; actor_dims; na]; head_init=actor_init, kwargs...)
        log_std = fill(Float32(log_std_init), na)
        ContinuousActor(act_net, log_std, rng, squash, log_min, log_max, action_clamp)
    else
        shared_net = mlp([input_size; actor_dims]; kwargs...)
        act_net = mlp([actor_dims; 2*na]; head_init=actor_init, kwargs...)
        log_std_net = mlp([actor_dims; 2*na]; head_init=actor_init, kwargs...)
        SDEActor(shared_net, act_net, log_std_net, rng, squash, log_min, log_max, action_clamp)
    end
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

function Actor(A::TupleSpace, input_size; shared_actor_dims=[], kwargs...)
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

function Critic(input_size; 
    critic_type = :scalar,
    critic_dims = [64, 64],
    critic_init = nothing,
    critic_loss_transform = identity,
    inv_critic_loss_transform = identity,
    categorical_values = nothing,
    kwargs...)

    if critic_type == :scalar
        head_init = @something critic_init Flux.orthogonal(; gain=1) error("Critic init error")
        last_dim = 1
        loss_transform = critic_loss_transform
        inv_loss_transform = inv_critic_loss_transform
        loss = Flux.mse
    elseif critic_type == :categorical
        @assert !isnothing(categorical_values) "Must provide categorical_values for critic_type = :categorical"
        head_init = @something critic_init Flux.orthogonal(; gain=0.01) error("Critic init error")
        last_dim = length(categorical_values)
        function inv_loss_transform(critic_out)
            transformed_value = reshape(categorical_values, 1, :) * softmax(critic_out; dims=1)
            inv_critic_loss_transform(transformed_value)
        end
        function loss_transform(value_target)
            transformed_target = critic_loss_transform(value_target)
            twohotbatch(transformed_target, categorical_values)
        end
        loss = Flux.logitcrossentropy
    else
        @assert false "critic_type = $critic_type is not supported"
    end
    net = mlp([input_size; critic_dims; last_dim]; head_init)
    Critic(net, loss, loss_transform, inv_loss_transform)
end

twohotbatch(x,r) = twohotbatch(Float32,x,r)
function twohotbatch(T::Type, x::AbstractArray{<:Real}, r::AbstractRange)
    @assert any(size(x) .== length(x)) "x must be a row or column vector"
    y = zeros(T, length(r), length(x))
    for (j, xj) in enumerate(x)
        i = findlast(_r->_r≤xj, r)
        if i == length(r)
            y[i,j] = 1
        elseif isnothing(i)
            y[1,j] = 1
        else
            x0, x1 = r[i], r[i+1]
            p = (x1 - xj) / (x1 - x0)
            y[i, j]   = p
            y[i+1, j] = 1-p
        end
    end
    y
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

## 
# Get action value information functions
##

@kwdef struct ACInput{O, A, AM}
    observation::O
    action::A       = nothing
    action_mask::AM = action isa Tuple ? (nothing for _ in 1:length(action)) : nothing
end
ACInput(O) = ACInput(observation = O)

struct PolicyOutput{A, AP, E}
    action::A
    log_prob::AP
    entropy::E
end

struct CriticOutput{A,B}
    value::A
    critic_out::B
end
CriticOutput(value) = CriticOutput(value, value)

function get_actionvalue(ac::ActorCritic, input::ACInput)
    shared_out = get_shared(ac.shared, input.observation)
    actor_input = ACInput(
        observation = shared_out,
        action = input.action,
        action_mask = input.action_mask
    )
    actor_out  = get_action(ac.actor, actor_input)
    critic_out = get_value(ac.critic, shared_out)
    return actor_out, critic_out
end

get_shared(f, x) = f(x)
get_shared(::Chain{Tuple{}}, x) = x

function get_value(critic::Critic, input)
    critic_out = critic.net(input)
    value = critic.inv_critic_loss_transform(critic_out)
    CriticOutput(value, critic_out)
end

function get_criticloss(critic::Critic, out::CriticOutput, value_target)
    target = @ignore_derivatives critic.critic_loss_transform(value_target)
    critic.loss(out.critic_out, target)
end

function get_action(ac::DiscreteActor, input::ACInput)
    actor_out = ac.actor(input.observation)

    if !isnothing(input.action_mask)
        actor_out += eltype(actor_out)(-1f10) * .!input.action_mask
    end

    log_probs = actor_out .- logsumexp(actor_out; dims=1)
    probs = softmax(actor_out; dims=1)

    entropy = -sum(log_probs .* probs, dims=1)

    if isnothing(input.action)
        action = sample_discrete.(ac.rng, eachcol(cpu(probs))) # need to implement for gpu?
    else
        action = input.action
    end

    idxs = CartesianIndex.(action, reshape(1:length(action), size(action)))
    action_log_prob = log_probs[idxs]

    return PolicyOutput(action, action_log_prob, entropy)
end

function get_action(ac::ContinuousActor, input::ACInput)
    action_mean = ac.actor(input.observation)
    log_std = clamp.(ac.log_std, ac.log_min, ac.log_max)
    commmon_continuous_action(ac, input, action_mean, log_std)
end

function get_action(ac::SDEActor, input::ACInput)
    shared = get_shared(ac.shared_net, input.observation)
    action_mean = ac.actor_net(shared)
    log_std = clamp.(ac.log_std_net(shared), ac.log_min, ac.log_max)
    commmon_continuous_action(ac, input, action_mean, log_std)
end

function commmon_continuous_action(ac::Union{ContinuousActor, SDEActor}, input::ACInput, action_mean, log_std)
    action_std = exp.(log_std)

    if isnothing(input.action)
        stand_normal = randn(ac.rng, eltype(action_mean), size(action_mean))
        action_normal = action_mean .+ action_std .* stand_normal

        action = if ac.squash
            tanh.(clamp.(action_normal, -ac.action_clamp, ac.action_clamp))
        else
            action_normal
        end
    else
        action = input.action
        action_normal = if ac.squash
            inv_tanh(action)
        else
            action
        end
        stand_normal = (action_normal .- action_mean) ./ action_std
    end

    if ac.squash
        action_log_prob = normal_logpdf(stand_normal, log_std) .- sum(log.(1 .- action .^ 2); dims=1)
        entropy = sum(-action_log_prob; dims=1) # estimate  
    else
        action_log_prob = normal_logpdf(stand_normal, log_std)
        entropy = sum(log_std; dims=1) .+ size(log_std,1) * (1+log(2f0*pi))/2
    end

    return PolicyOutput(action, action_log_prob, entropy)
end

function get_action(actor::TupleActor, input::ACInput)
    shared_out = get_shared(actor.shared, input.observation)

    N = length(actor.actors)
    action_tuples = ACInput.(_to_tuple.((shared_out, input.action, input.action_mask), N)...)

    action_info = get_action.(actor.actors, action_tuples)

    return PolicyOutput(
        Tuple(info.action   for info in action_info), 
        Tuple(info.log_prob for info in action_info), 
        Tuple(info.entropy  for info in action_info)
    )
end
_to_tuple(x::Tuple, N) = x
_to_tuple(x, N) = Tuple(x for _ in 1:N)

function (ac::ActorCritic)(state)
    actor_out, critic_out = get_actionvalue(ac, ACInput(state))
    return actor_out.action
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