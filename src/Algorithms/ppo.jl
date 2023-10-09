@with_kw struct PPOSolver{
    E<:AbstractMultiEnv, 
    RNG<:AbstractRNG, 
    OPT<:Flux.Optimisers.AbstractRule, 
    AC<:ActorCritic
    }
    
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
    opt = Flux.setup(opt_0, ac) |> device

    buffer = Buffer(env, traj_len) |> device
    flat_buffer = fmap(x->reshape(x, size(x)[1:end-2]..., :), buffer; exclude=x->x isa AbstractArray) # points to same data as buffer
    mini_buffer = fmap(x->similar(x, size(x)[1:end-1]..., batch_size), flat_buffer; exclude=x->x isa AbstractArray)

    solve(solver, ac, opt, buffer, flat_buffer, mini_buffer)
end

function solve(solver::PPOSolver, ac, opt, buffer, flat_buffer, mini_buffer)
    @assert solver.device == cpu # temp for development

    @unpack env, n_steps, discount, gae_lambda, lr, lr_decay, max_time = solver

    start_time = time()
    info = Logger{Tuple{Vector, Vector}}()

    reset!(env)

    n_transitions = length(env) * buffer.traj_len
    
    prog = Progress(floor(n_steps / n_transitions) |> Int)

    for global_step in n_transitions:n_transitions:n_steps
        fill_buffer!(env, buffer, ac)

        gae!(buffer, discount, gae_lambda)

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

    return cpu(ac), info.log
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

    for idxs in Iterators.partition(randperm(flat_buffer.traj_len), batch_size)
        sample_batch!(mini_buffer, flat_buffer, idxs)

        @unpack s, a, a_logprob, returns, action_mask = mini_buffer

        advantages = if norm_advantages
            normalize(mini_buffer.advantages)
        else
            mini_buffer.advantages
        end

        grads = Flux.gradient(ac) do ac
            (_, newlogprob, entropy, newvalue) = get_actionvalue(ac, s, a; action_mask)

            log_ratio = newlogprob .- a_logprob
            ratio = exp.(log_ratio)
            pg_loss1 = -advantages .* ratio
            pg_loss2 = -advantages .* clamp.(ratio, 1-clip_coef, 1+clip_coef)
            policy_loss = mean(max.(pg_loss1, pg_loss2))
        
            value_loss = Flux.mse(newvalue, returns)
            entropy_loss = mean(entropy)
            total_loss = policy_loss - ent_coef*entropy_loss + vf_coef*value_loss

            ignore_derivatives() do 
                clip_frac = mean(abs.(ratio .- 1) .> clip_coef)
                kl_est = mean((ratio .- 1) .- log_ratio)
    
                loss_info(; 
                    policy_loss, value_loss, entropy_loss, total_loss, clip_frac, kl_est, ac.log_std
                ) 
            end

            return total_loss
        end

        Flux.update!(opt, ac, grads[1])
    end

    return map(mean, (; loss_info.log...))
end

normalize(x::AbstractVector, eps = eltype(x)(1e-8)) = (x .- mean(x)) / (std(x) + eps)
