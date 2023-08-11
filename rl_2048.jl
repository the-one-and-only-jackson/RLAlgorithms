using CommonRLInterface
using CommonRLInterface.Wrappers: QuickWrapper

using Parameters: @with_kw
using Random: default_rng, AbstractRNG
using StaticArrays: SA, @SArray, SArray
using Statistics: mean, std
using Flux

using RLAlgorithms.Spaces: Discrete, Box, NumericArraySpace, SpaceStyle, UnknownSpaceStyle
using RLAlgorithms.MultiEnv: VecEnv
using RLAlgorithms.PPO
using RLAlgorithms.MultiEnvWrappers
using RLAlgorithms.ActorCritics: get_actionvalue, DiscreteActorCritic

const RL = CommonRLInterface

const My2048_actions = SA[:up, :down, :left, :right]

@with_kw mutable struct My2048 <: AbstractEnv
    board::Matrix{UInt16} = zeros(UInt16, (4,4))
    valid_action_mask::Vector{Bool} = trues(4)
    done::Bool = false
    score::UInt16 = 0
    high_score::UInt16 = 0
    rng::AbstractRNG = default_rng()
end

RL.actions(::My2048) = Discrete(4)

RL.observations(::My2048) = Box{Bool}(SArray{Tuple{4,4,16}}(falses(4,4,16)), SArray{Tuple{4,4,16}}(trues(4,4,16)))

function RL.observe(env::My2048)
    o = BitArray(undef,(4,4,16))
    for k = 1:16
        check_val = 0x0001 << k
        for i=1:4, j=1:4
            o[i,j,k] = env.board[i,j] == check_val
        end
    end
    return o
end

RL.terminated(env::My2048) = env.done
RL.valid_action_mask(env::My2048) = env.valid_action_mask
RL.valid_actions(env::My2048) = actions(env)[RL.valid_action_mask(env)]

function RL.reset!(env::My2048)
    env.board .= 0
    env.score = 0
    gen_square!(env.rng, env.board)
    gen_square!(env.rng, env.board)
    update_valid_actions!(env)
    nothing
end

function RL.act!(env::My2048, a::Integer)
    if !valid_action_mask(env)[a]
        println("Invalid action")
        return -1
    end
    act!(env, My2048_actions[a])
end
function RL.act!(env::My2048, a::Symbol)
    iter_fun = (a==:up || a==:down) ? eachcol : eachrow
    points = sort_and_combine!.(iter_fun(env.board), a==:down || a==:right) |> sum

    env.score += points
    env.high_score = max(env.score, env.high_score)

    env.done = gen_square!(env.rng, env.board)
    update_valid_actions!(env)

    return points
end

function sort_and_combine!(v, reverse=false)
    reverse && reverse!(v)

    i = 0
    combine_flag = true
    score = 0
    for x in v
        iszero(x) && continue
        combine_flag = !combine_flag && v[i] == x
        v[i += !combine_flag] = (1 + combine_flag) * x
        score += v[i] * combine_flag
    end
    v[i+1:end] .= 0

    reverse && reverse!(v)

    return score
end

function gen_square!(rng, board) # return true if board full
    idx_vec = findall(iszero, board)
    isempty(idx_vec) && return true
    board[rand(rng, idx_vec)] = rand(rng) < 0.9 ? 2 : 4
    return false
end

function update_valid_actions!(env::My2048)
    for (ii,iter_fun) in enumerate((eachcol, eachrow))
        updown = can_move_combine.(iter_fun(env.board))
        env.valid_action_mask[(1:2) .+ 2*(ii-1)] .= (any(x->x[j], updown) for j in 1:2)
    end
    env.done |= iszero(env.valid_action_mask)
    nothing
end

function can_move_combine(v)
    for ii = 1:length(v)-1
        if !iszero(v[ii]) && v[ii] == v[ii+1]
            return (true, true)
        end
    end
    
    vals = iszero.(v)
    flag1, flag2 = false, false
    if 0 < count(vals) < length(vals)
        flag1 = findfirst(vals) < findlast(.!vals) # can move up (first empty before last non-empty square)
        flag2 = findfirst(.!vals) < findlast(vals) # can move down (first non-empty square before last empty square)
    end
    return (flag1, flag2)
end

function evaluate(env, policy; max_steps=10_000)
    RL.reset!(env)
    r = 0.0
    for _ in 1:max_steps
        o = observe(env)
        a = policy(env, o)
        r += act!(env, a)
        if terminated(env)
            break
        end
    end
    return r
end

function conv_ac()
    hidden_init = Flux.orthogonal(; gain=sqrt(2))
    actor_init  = Flux.orthogonal(; gain=0.01)
    critic_init = Flux.orthogonal(; gain=1)

    shared = Chain(
        Conv((2,2), 16=>16, relu; init=hidden_init),
        Conv((2,2), 16=>16, relu; init=hidden_init),
        Flux.flatten
    )

    sz = Flux.outputsize(shared, (4,4,16,1))

    actor = Chain(Dense(sz[1]=>4; init=actor_init))
    critic = Chain(Dense(sz[1]=>1; init=critic_init))

    DiscreteActorCritic(shared,actor,critic,default_rng())
end

# transform_rule(x) = (relu(log2(x)) - 8)/8
transform_rule(x) = @. x |> Float32 |> log2 |> relu |> y->y/11

vec_env = VecEnv(()->My2048(), n_envs=8) |> LoggingWrapper

env = TransformWrapper(; 
    env = vec_env,  
    action_fun = transform_rule
)

# AC: assert promote type

solver = PPOSolver(;
    env,
    n_steps = 50_000,
    lr = 3f-4,
    lr_decay = false,
    vf_coef = 0.5,
    traj_len = 128,
    batch_size = 128,
    n_epochs = 4,
    discount = 0.99,
    gae_lambda = 0.95f0, # 1.0 corresponds to not using GAE
    norm_advantages = true,
    kl_targ = 0.02,
    ac = conv_ac()
)

ac, info = solve(solver)




using Plots
plot(vec_env.step, vec_env.reward)
plot(vec_env.step, vec_env.episode_length)
plot(vec_env.step, 1:length(vec_env.step))


plot(info[:entropy_loss][1], info[:entropy_loss][2])
plot(info[:kl_est][1], info[:kl_est][2])



function policy(env,o)
    (action, _, _, _) = get_actionvalue(ac, o; action_mask = valid_action_mask(env)) 
    return action[]
end

evaluate(env, policy)


baseline_policy(env, o) = (1:4)[valid_action_mask(env)] |> rand
temp = [evaluate(My2048(), baseline_policy) for _ in 1:100]
mean(temp)
std(temp)/sqrt(length(temp))




conv_ac()




