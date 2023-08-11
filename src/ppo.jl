module PPO

using CommonRLInterface, Flux, ProgressMeter, Parameters
using CommonRLInterface.Wrappers: QuickWrapper

using Random: default_rng, seed!, randperm
using Statistics: mean, std
using ChainRules: ignore_derivatives, @ignore_derivatives
using LinearAlgebra: norm

using ..Utils
using ..ActorCritics
using ..Spaces
using ..MultiEnv

export 
    solve,
    PPOSolver

@with_kw struct Buffer
    s
    a
    r = zeros(Float32, size(s)[end-1:end])
    done = zeros(Bool, size(s)[end-1:end])
    a_logprob = copy(r)
    a_mask = nothing
    value = copy(r)
    advantages = copy(r)
    returns = copy(r)
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
    end

    if provided(valid_action_mask, env)
        a_mask = trues(size(A)..., n_envs, traj_len)
        return Buffer(; s, a, a_mask)
    else
        return Buffer(; s, a)
    end
end

Flux.@functor Buffer

function sample_batch!(dst::T, src::T, idxs) where {T <: Buffer}
    copy_fun(x, y) = copyto!(x, selectdim(y, ndims(y), idxs))
    fmap(copy_fun, dst, src)
end

@with_kw struct PPOSolver
    env::AbstractMultiEnv
    n_steps::Int64 = 1_000_000
    lr::Float32 = 3f-4
    lr_decay::Bool = true
    traj_len::Int64 = 2048
    batch_size::Int64 = 64
    n_epochs::Int64 = 10
    discount::Float32 = 0.99
    gae_lambda::Float32 = 0.95
    clip_coef::Float32 = 0.5
    norm_advantages::Bool = true
    ent_coef::Float32 = 0
    vf_coef::Float32 = 0.5
    rng = default_rng()
    kl_targ::Float32 = 0.02
    device::Function = cpu # cpu or gpu
    opt_0 = Flux.Optimisers.Adam(lr)
    ac::ActorCritic = ActorCritic(env) 
    
    @assert device in [cpu, gpu]
    @assert iszero(Int64(length(env)*traj_len)%batch_size)
end

function solve(solver::PPOSolver)
    @unpack_PPOSolver solver

    start_time = time()
    info = Logger{Tuple{Vector, Vector}}()

    reset!(env)

    buffer = Buffer( env, traj_len)

    buffer, ac = (buffer, ac) .|> device
    opt = Flux.setup(opt_0, ac)

    flat_buffer = fmap(x->reshape(x, size(x)[1:end-2]..., :), buffer) # points to same data as buffer
    mini_buffer = fmap(x->similar(x, size(x)[1:end-1]..., batch_size), flat_buffer)

    n_transitions = Int(length(env)*traj_len)
    prog = Progress(floor(n_steps / n_transitions) |> Int)
    for global_step in n_transitions:n_transitions:n_steps
        for traj_step in 1:traj_len
            rollout!(env, buffer, ac, traj_step, device)
        end

        (_, _, _, last_value) = get_actionvalue(ac, observe(env) |> stack |> device)    
        gae!(buffer, last_value, discount, gae_lambda)

        learning_rate = lr_decay ? (1-global_step/n_steps)*lr : lr
        Flux.Optimisers.adjust!(opt, learning_rate)

        for epoch in 1:n_epochs
            loss_info = train_batch!(ac, opt, flat_buffer, mini_buffer, solver)

            if epoch == n_epochs || loss_info.kl_est > kl_targ
                info(global_step; loss_info...) # this logging may be slightly inaccurate
                break
            end
        end
        
        info(global_step; learning_rate, wall_time = time() - start_time)

        showvalues = zip(keys(info.log), map(x->x[2][end], values(info.log)))
        next!(prog; showvalues)
    end
    finish!(prog)

    return cpu(ac), info.log
end

function rollout!(env, buffer, ac, traj_step, device)
    s = observe(env) |> stack |> device
    a_mask = provided(valid_action_mask, env) ? device(stack(valid_action_mask(env))) : nothing
    (a, a_logprob, _, value) = get_actionvalue(ac, s; action_mask = a_mask)

    idxs = CartesianIndex.(a, 1:length(a))
    if any(.!a_mask[idxs])
        println(a)
        println(a_mask)
        println(a_logprob)
        @assert false
    end

    r = act!(env, cpu(a))
    done = terminated(env)
    send_to!(buffer, traj_step; s, a, a_logprob, value, r, done, a_mask)
    nothing
end

function send_to!(buffer::Buffer, idx::Int; kwargs...)
    for (key, val) in kwargs
        isnothing(val) && continue
        arr = getfield(buffer, key)
        copyto!(selectdim(arr, ndims(arr), idx), val)
    end
    nothing
end

function gae!(buffer::Buffer, last_value, discount, gae_lambda)
    @unpack r, done, value, advantages, returns = buffer
    last_advantage = fill!(similar(last_value), zero(eltype(last_value)))
    for (advantages, r, done, value) in Iterators.reverse(zip(eachcol.((advantages, r, done, value))...))
        td = @. r + discount * !done * last_value - value
        @. advantages = td + gae_lambda * discount * !done * last_advantage
        last_value, last_advantage = value, advantages
    end
    @. returns = advantages + value
    nothing
end

function train_batch!(
    ac::ActorCritic,
    opt,
    flat_buffer::Buffer, 
    mini_buffer::Buffer,
    solver::PPOSolver
    )

    @unpack batch_size, norm_advantages, clip_coef, ent_coef, vf_coef = solver

    loss_info = Logger{Vector}()

    N = size(flat_buffer.done,1)
    for idxs in Iterators.partition(randperm(N), batch_size)
        sample_batch!(mini_buffer, flat_buffer, idxs)

        @unpack s, a, a_logprob, advantages, returns, a_mask = mini_buffer
        if norm_advantages
            advantages = normalize(mini_buffer.advantages)
        end

        grads = Flux.gradient(ac) do ac
            (_, newlogprob, entropy, newvalue) = get_actionvalue(ac, s; action=vec(a), action_mask=a_mask)
            (policy_loss, clip_frac, kl_est) = policy_loss_fun(a_logprob, newlogprob, advantages, clip_coef)
            value_loss = Flux.mse(newvalue, returns)/2
            entropy_loss = mean(entropy)
            total_loss = policy_loss - ent_coef*entropy_loss + vf_coef*value_loss

            @ignore_derivatives loss_info(; 
                policy_loss, value_loss, entropy_loss, total_loss, clip_frac, kl_est
            )

            return total_loss
        end

        Flux.update!(opt, ac, clip_grads(grads)[1])
    end

    return map(mean, (; loss_info.log...))
end

function policy_loss_fun(
    a_logprob::T, 
    newlogprob::T, 
    advantages::T, 
    clip_coef::T2
    ) where {T2<:Real, T<:AbstractVector{T2}}

    log_ratio = newlogprob .- a_logprob
    ratio = exp.(log_ratio)
    pg_loss1 = -advantages .* ratio
    pg_loss2 = -advantages .* clamp.(ratio, 1-clip_coef, 1+clip_coef)
    policy_batch_loss = mean(max.(pg_loss1, pg_loss2))

    clip_fracs = @ignore_derivatives mean(abs.(ratio .- 1) .> clip_coef)
    kl_est = @ignore_derivatives mean((ratio .- 1) .- log_ratio)

    return policy_batch_loss, clip_fracs, kl_est
end

function clip_grads(grads, thresh=0.5f0)
    isinf(thresh) && return grads
    grad_vec, model = Flux.Optimisers.destructure(grads)
    ratio = thresh / norm(grad_vec)
    if ratio < 1
        grad_vec .*= ratio
    end
    clip_grads = model(grad_vec)
end

end