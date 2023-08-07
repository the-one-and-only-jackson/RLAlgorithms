module PPO

using CommonRLInterface, Flux, ProgressMeter, Parameters
using CommonRLInterface.Wrappers: QuickWrapper

using Random: default_rng, seed!, randperm
using Statistics: mean, std
using ChainRules: ignore_derivatives, @ignore_derivatives

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
    value = copy(r)
    advantages = copy(r)
    returns = copy(r)
end
function Buffer(env::AbstractMultiEnv, traj_len::Int)
    n_envs = length(env)
    s_sz = size(single_observations(env))
    a_sz = size(single_actions(env))
    a_T = eltype(single_actions(env))

    s = zeros(Float32, s_sz..., n_envs, traj_len)
    a = zeros(a_T, a_sz..., n_envs, traj_len)

    return Buffer(; s, a)
end

Flux.@functor Buffer

function sample_batch!(dst::Buffer, src::Buffer, idxs)
    copy_fun(x, y) = copyto!(x, selectdim(y, ndims(y), idxs))
    fmap(copy_fun, dst, src)
end

function send_to!(buffer::Buffer, idx::Int; kwargs...)
    for (key, val) in kwargs
        arr = getfield(buffer, key)
        copyto!(selectdim(arr, ndims(arr), idx), val)
    end
    nothing
end

@with_kw struct Logger{T}
    log = Dict{Symbol, T}()
end
function (logger::Logger{Tuple{Vector, Vector}})(x_val; kwargs...)
    for (key, val) in kwargs
        x_vec, y_vec = get!(logger.log, key, (typeof(x_val)[], typeof(val)[]))
        push!(x_vec, x_val)
        push!(y_vec, val)
    end
    nothing
end
function (logger::Logger{Vector})(; kwargs...)
    for (key, val) in kwargs
        y_vec = get!(logger.log, key, typeof(val)[])
        push!(y_vec, val)
    end
    nothing
end

@with_kw struct PPOSolver
    env::AbstractMultiEnv
    n_steps::Int64 = 1_000_000
    lr::Float32 = 3f-4
    lr_decay::Bool = true
    traj_len::Int64 = 2048
    batch_size::Int64 = 64
    n_epochs::Int64 = 10
    discount::Float32 = 0.99f0
    gae_lambda::Float32 = 1f0 # 1.0 corresponds to not using GAE
    clip_coef::Float32 = 0.5f0
    norm_advantages::Bool = true
    ent_coef::Float32 = 0
    vf_coef::Float32 = 0.5f0
    rng = default_rng()
    seed::Int64 = 0
    device::Function = cpu # cpu or gpu
    kl_targ::Float64 = 0.02
    opt_0 = Flux.Optimisers.Adam(lr)
    ac::ActorCritic = ActorCritic(env) 
    
    @assert device in [cpu, gpu]
    @assert iszero(n_transitions%batch_size) "traj_len*n_envs not divisible by batch_size"
end

function solve(solver::PPOSolver)
    @unpack_PPOSolver solver

    n_transitions = Int64(length(env)*traj_len)

    start_time = time()
    info = Logger{Tuple{Vector, Vector}}()

    seed!(rng, seed)
    reset!(env)

    buffer, ac = (Buffer(; env, traj_len), ac) .|> device
    opt = Flux.setup(opt_0, ac)

    # used for training
    flat_buffer = fmap(x->reshape(x, size(x)[1:end-2]..., :), buffer)
    mini_buffer = fmap(x->similar(x, size(x)[1:end-1]..., batch_size), flat_buffer)

    prog = Progress(n_steps)
    for global_step in n_transitions:n_transitions:n_steps
        last_value = rollout!(env, buffer, ac, traj_len, device)
        gae!(buffer, last_value, discount, gae_lambda)

        learning_rate = lr_decay ? (1-global_step/n_steps)*lr : lr
        Flux.Optimisers.adjust!(opt, learning_rate)

        for epoch in 1:n_epochs
            loss_info = train_batch!(ac, opt, flat_buffer, mini_buffer, solver)

            if epoch == n_epochs || loss_info.kl_est > kl_targ
                info(global_step; loss_info...)
                break
            end
        end
        
        info(global_step; learning_rate, wall_time = time() - start_time)

        next!(prog; 
            step = n_transitions, 
            showvalues = zip(keys(info.log), map(x->x[2][end], values(info.log)))
        )
    end
    finish!(prog)

    return cpu(ac), info.log
end

function rollout!(env, buffer, ac, traj_len, device)
    for traj_step in 1:traj_len
        s = observe(env) |> stack .|> Float32 |> device
        (a, a_logprob, _, value) = get_actionvalue(ac, s)
        r = act!(env, cpu(a))
        done = terminated(env)
        send_to!(buffer, traj_step; s, a, a_logprob, value, r, done)
    end
    last_s = observe(env) |> stack .|> Float32 |> device
    (_, _, _, last_value) = get_actionvalue(ac, last_s)
    return last_value
end

function gae!(buffer::Buffer, last_value, discount, gae_lambda)
    last_advantage = similar(buffer.advantages, size(buffer.advantages,1)) .= 0
    itrs = eachcol.((buffer.advantages, buffer.r, buffer.done, buffer.value))
    for (advantages, r, done, value) in Iterators.reverse(zip(itrs...))
        td = r .+ discount * (.!done .* last_value) .- value
        advantages .= td .+ (discount*gae_lambda) * (.!done .* last_advantage)
        last_value, last_advantage = value, advantages
    end
    buffer.returns .= buffer.advantages .+ buffer.value
    nothing
end

function train_batch!(
        ac::ActorCritic,
        opt,
        flat_buffer::Buffer, 
        mini_buffer::Buffer,
        solver::PPOSolver
    )

    @unpack_PPOSolver solver

    loss_info = Logger{Vector}()

    for idxs in Iterators.partition(randperm(size(flat_buffer.done,1)), batch_size)
        sample_batch!(mini_b, flat_buffer, idxs)

        @unpack s, a, a_logprob, advantages = mini_buffer
        if norm_advantages
            advantages = normalize(mini_buffer.advantages)
        end

        grads = Flux.gradient(ac) do ac
            (_, newlogprob, entropy, newvalue) = get_actionvalue(ac, s, a)
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
    l2 = norm(grad_vec)
    if l2 > thresh
        grad_vec *= (thresh / l2)
    end
    clip_grads = model(grad_vec)
end

normalize(x; eps = 1f-8) = (x .- mean(x)) / (std(x) + eps)

end