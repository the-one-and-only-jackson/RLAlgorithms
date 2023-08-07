using CommonRLInterface
using Parameters: @with_kw
using Random: default_rng

const RL = CommonRLInterface

@with_kw mutable struct My2048 <: AbstractEnv
    board = zeros(UInt16, (4,4))
    valid_action_mask = trues(4)
    done = false
    score = 0
    high_score = 0
    rng = default_rng()
end

RL.observe(env::My2048) = env.board
RL.terminated(env::My2048) = env.done
RL.actions(::My2048) = [:up, :down, :left, :right]
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

function RL.act!(env::My2048, a::Symbol)
    # a=1,2 => up/down, a=3,4 => left/right
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
    v[i+1:length(v)] .= 0

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

using CommonRLInterface.Wrappers: QuickWrapper

QuickWrapper(
    env;
    actions = 1:4,
    # observations = 
    observe = env -> reshape(observe(env) |> log2, :),
    act! = (env,a) -> act!(env, actions(env)[a])
)

# next:
# wrapper to standardize actions/observations + spaces
# ppo with discrete + action masking (action masking coming from actor critic, not ppo)



env = My2048()
RL.reset!(env)
@time for step in 1:1000
    a = rand(valid_actions(env))
    act!(env, a)
    if terminated(env)
        println(step)
        break
    end
end

a = rand(valid_actions(env))
act!(env, a)
observe(env)


update_valid_actions!(env)
env.valid_action_mask
env.done