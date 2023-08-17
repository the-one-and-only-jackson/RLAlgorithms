module GameEnv

export My2048

using CommonRLInterface
using Parameters: @with_kw
using Random: default_rng, AbstractRNG
using StaticArrays: SA, @SArray, SArray
using RLAlgorithms.Spaces: Discrete, Box

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

RL.observe(env::My2048) = copy(env.board)
# function RL.observe(env::My2048)
#     o = BitArray(undef,(4,4,16))
#     for k = 1:16
#         check_val = 0x0001 << k
#         for i=1:4, j=1:4
#             o[i,j,k] = env.board[i,j] == check_val
#         end
#     end
#     return o
# end

RL.terminated(env::My2048) = env.done
RL.valid_action_mask(env::My2048) = env.valid_action_mask
RL.valid_actions(env::My2048) = actions(env)[RL.valid_action_mask(env)]

function RL.reset!(env::My2048)
    env.board .= 0
    env.score = 0
    env.done = false
    gen_square!(env.rng, env.board)
    gen_square!(env.rng, env.board)
    update_valid_actions!(env)
    nothing
end

RL.act!(env::My2048, a::Integer) = act!(env, My2048_actions[a])

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

end