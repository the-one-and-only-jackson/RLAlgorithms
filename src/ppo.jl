module PPO

using CommonRLInterface, Flux, ProgressMeter, Parameters
using CommonRLInterface.Wrappers: QuickWrapper

using Random: AbstractRNG, default_rng, seed!, randperm
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
    a_logprob::R = copy(r)
    a_mask::AM = nothing
    value::R = copy(r)
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
        a_mask = trues(size(A)..., n_envs, traj_len)
        println("Action mask provided.")
    else
        a_mask = nothing
    end

    return Buffer(; s, a, a_mask, traj_len)
end

Flux.@functor Buffer

function sample_batch!(dst::T, src::T, idxs) where {T <: Buffer}
    copy_fun(x, y) = copyto!(x, selectdim(y, ndims(y), idxs))
    fmap(copy_fun, dst, src; exclude=x->x isa AbstractArray)
    nothing
end

@with_kw struct PPOSolver{E<:AbstractMultiEnv, RNG<:AbstractRNG, OPT<:Flux.Optimisers.AbstractRule, AC<:ActorCritic}
    env::E
    n_steps::Int64 = 1_000_000
    max_time::Int = 60*10 # 10 min timer
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
    rng::RNG = default_rng()
    kl_targ::Float32 = 0.02
    device::Function = cpu # cpu or gpu
    opt_0::OPT = Flux.Optimisers.Adam(lr)
    ac::AC = ActorCritic(env) 
    
    @assert device in [cpu, gpu]
    @assert iszero(Int64(length(env)*traj_len)%batch_size)
end

function solve(solver::PPOSolver)
    @unpack_PPOSolver solver

    ac = ac |> device
    opt = Flux.setup(opt_0, ac)

    buffer = Buffer(env, traj_len) |> device
    flat_buffer = fmap(x->reshape(x, size(x)[1:end-2]..., :), buffer; exclude=x->x isa AbstractArray) # points to same data as buffer
    mini_buffer = fmap(x->similar(x, size(x)[1:end-1]..., batch_size), flat_buffer; exclude=x->x isa AbstractArray)

    solve(solver, ac, opt, buffer, flat_buffer, mini_buffer)
end

function solve(solver::PPOSolver, ac, opt, buffer, flat_buffer, mini_buffer)
    @assert solver.device == cpu
    @unpack env, n_steps, discount, gae_lambda, lr, lr_decay, max_time = solver

    start_time = time()
    info = Logger{Tuple{Vector, Vector}}()

    reset!(env)

    n_transitions = length(env) * buffer.traj_len
    prog = Progress(floor(n_steps / n_transitions) |> Int)
    for global_step in n_transitions:n_transitions:n_steps
        # fill_buffer!(env, buffer, ac, device)
        fill_buffer!(env, buffer, ac)

        # (_, _, _, last_value) = get_actionvalue(ac, observe(env) |> stack |> device)
        (_, _, _, last_value) = get_actionvalue(ac, observe(env) |> stack)    

        gae!(buffer, last_value, discount, gae_lambda)

        learning_rate = lr_decay ? (lr - lr*global_step/n_steps) : lr
        Flux.Optimisers.adjust!(opt, learning_rate)

        loss_info = train_epochs!(ac, opt, flat_buffer, mini_buffer, solver)
        
        info(global_step; wall_time = time() - start_time, learning_rate, loss_info...)

        if time() - start_time > max_time
            break
        end

        next!(prog; showvalues = zip(keys(info.log), map(x->x[2][end], values(info.log))))
    end
    finish!(prog)

    # return cpu(ac), info.log
    return ac, info.log
end

# function fill_buffer!(env::AbstractMultiEnv, buffer::Buffer, ac::ActorCritic, device)
#     to_device(x) = x |> stack |> device
#     for _ in 1:buffer.traj_len
#         s = observe(env) |> to_device
#         a_mask = ifelse(provided(valid_action_mask,env), valid_action_mask(env) |> to_device, nothing)
#         (a, a_logprob, _, value) = get_actionvalue(ac, s; action_mask = a_mask)
#         r = act!(env, cpu(a))
#         done = terminated(env)
#         send_to!(buffer; s, a, a_logprob, value, r, done, a_mask)
#     end
#     nothing
# end

function fill_buffer!(env::AbstractMultiEnv, buffer::Buffer, ac::ActorCritic)
    for _ in 1:buffer.traj_len
        s = observe(env) |> stack

        a_mask = if provided(valid_action_mask,env)
            valid_action_mask(env) |> stack
        else
            nothing
        end
        
        (a, a_logprob, _, value) = get_actionvalue(ac, s; action_mask = a_mask)
        r = act!(env, a)
        done = terminated(env)
        send_to!(buffer; s, a, a_logprob, value, r, done, a_mask)
    end
    nothing
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

function train_epochs!(ac, opt, flat_buffer, mini_buffer, solver)
    loss_info = nothing
    for _ in 1:solver.n_epochs
        loss_info = train_batch!(ac, opt, flat_buffer, mini_buffer, solver)
        loss_info.kl_est > solver.kl_targ && break
    end
    return loss_info
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

    N = length(flat_buffer.done)
    for idxs in Iterators.partition(randperm(N), batch_size)
        sample_batch!(mini_buffer, flat_buffer, idxs)

        @unpack s, a, a_logprob, returns, a_mask = mini_buffer

        if norm_advantages
            advantages = normalize(mini_buffer.advantages)
        else
            advantages = mini_buffer.advantages
        end

        grads = Flux.gradient(ac) do ac
            (_, newlogprob, entropy, newvalue) = get_actionvalue(ac, s, a; action_mask=a_mask)
            (policy_loss, clip_frac, kl_est) = policy_loss_fun(a_logprob, newlogprob, advantages, clip_coef)
            value_loss = Flux.mse(newvalue, returns)/2
            entropy_loss = mean(entropy)
            total_loss = policy_loss - ent_coef*entropy_loss + vf_coef*value_loss

            @ignore_derivatives loss_info(; 
                policy_loss, value_loss, entropy_loss, total_loss, clip_frac, kl_est
            )

            return total_loss
        end

        # Flux.update!(opt, ac, clip_grads(grads)[1])
        Flux.update!(opt, ac, grads[1])
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