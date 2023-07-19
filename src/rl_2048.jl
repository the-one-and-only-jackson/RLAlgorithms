using CommonRLInterface
using Parameters: @with_kw 

const RL = CommonRLInterface

function shift!(v)
    N = length(v)
    i = 1
    for j = 1:N
        if !iszero(v[j])
            v[i] = v[j]
            i += 1
        end
    end
    v[i:N] .= 0
    nothing
end

function combine!(v)
    score = 0
    i, j = 1, 1
    while i < length(v)
        if v[i] == v[i+1]
            v[j] = 2*v[i]
            score += v[j]
            j += 1
            i += 2
        else
            v[j] = v[i]
            j += 1
            i += 1
        end
    end

    if i == length(v)
        v[j] = v[i]
        j += 1
    end
    
    v[j:end] .= 0

    return score
end

@with_kw struct My2048 <: AbstractEnv
    board::Matrix{UInt16} = zeros(UInt16, (4,4))
    done::Vector{Bool} = [false]
end

function RL.act!(env::My2048, a::Integer)
    @assert a>0 && a<5 "action out of range"

    if a > 2
        a -= 2
        reverse!(env.board, dims=a)
        rev = true
    else
        rev = false
    end

    vecs = a==1 ? eachcol(env.board) : eachrow(env.board)
    shift!.(vecs)
    score = sum(combine!.(vecs))

    env.done[] = gen_square!(env.board)

    if rev
        reverse!(env.board, dims=a)
    end

    return score
end

function gen_square!(board)
    idx_vec = findall(iszero, board)
    isempty(idx_vec) && return true
    board[rand(idx_vec)] = rand() < 0.9 ? 2 : 4
    return false
end

function RL.reset!(env::My2048)
    env.board .= 0
    gen_square!(env.board)
    gen_square!(env.board)
    nothing
end

env = My2048()
RL.reset!(env)
env.board
RL.act!(env, 1)
env.board

# ah shit, needs action masking! 