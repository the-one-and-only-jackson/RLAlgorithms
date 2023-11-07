module Bitboard2048

export My2048, get_rand_valid_action

using Parameters: @with_kw
using CommonRLInterface
using Random: AbstractRNG, default_rng

@with_kw mutable struct My2048{T<:AbstractRNG} <: AbstractEnv
    board::UInt64 = 0
    valid_action_mask::BitVector = trues(4)
    done::Bool = false
    score::UInt64 = 0
    high_score::UInt64 = 0
    rng::T = default_rng()
    biggest_tile::UInt8 = 0
end

const RL = CommonRLInterface

function RL.reset!(env::My2048)
    env.biggest_tile = max(env.biggest_tile, max_tile(env.board))
    env.board = add_square(env.rng, add_square(env.rng, zero(UInt64)))
    env.valid_action_mask.chunks[] = action_check(env.board)
    env.done = false
    env.score = 0
    nothing
end

RL.observe(env::My2048) = env.board
RL.terminated(env::My2048) = env.done
RL.valid_action_mask(env::My2048) = env.valid_action_mask

function RL.act!(env::My2048, a::Integer)
    if !env.valid_action_mask[a] || a<1 || a>4
        return zero(env.score)
    end

    if a == 1
        (env.board, points) = move_right(env.board)
    elseif a == 2
        (env.board, points) = move_left(env.board)
    elseif a == 3
        (env.board, points) = move_down(env.board)
    elseif a == 4
        (env.board, points) = move_up(env.board)
    end

    env.score += points
    env.high_score = max(env.high_score, env.score)

    env.board = add_square(env.rng, env.board)

    env.valid_action_mask.chunks[] = action_check(env.board)
    env.done = iszero(env.valid_action_mask.chunks[])

    return points
end

function max_tile(x::UInt64)
    y = zero(x)
    for i in 0:4:60
        val = (x >> i) & 0xf
        y = max(y, val)
    end
    return y
end

function prettyprint(x::UInt64)
    board = Matrix{Int}(undef,(4,4))
    for i in 0:15
        pow2 = (x >> (4*i)) & 0xf
        board[i+1] = iszero(pow2) ? 0 : 1 << pow2
    end
    display(board)
    nothing
end

move_up(x::UInt64) = move_up_down(x; down=false)
move_down(x::UInt64) = move_up_down(x; down=true)
function move_up_down(x::UInt64; down=false)
    itr = down ? (12:-4:0) : (0:4:12)
    y = zero(UInt64)
    score = zero(UInt64)
    for j in 0:16:48
        indicator = UInt64(0xf) << j
        shift = first(itr) - itr.step
        can_add = false
        last_val = 0
        for i in itr
            v = (x >> i) & indicator
            iszero(v) && continue
            can_add = !can_add || v != last_val
            if can_add
                shift += itr.step
                last_val = v
            else
                score += 2 << (last_val >> j)
                last_val = 1 << j
            end
            y += last_val << shift
        end
    end
    return y, score
end

move_left(x) = move_left_right(x; right=false)
move_right(x) = move_left_right(x; right=true)
function move_left_right(x::UInt64; right=false)
    itr = right ? (48:-16:0) : (0:16:48)
    y = zero(UInt64)
    score = zero(UInt64)
    for j in 0:4:12
        indicator = UInt64(0xf) << j
        shift = first(itr) - itr.step
        can_add = false
        last_val = 0
        for i in itr
            v = (x >> i) & indicator
            iszero(v) && continue
            can_add = !can_add || v != last_val
            if can_add
                shift += itr.step
                last_val = v
            else
                score += 2 << (last_val >> j)
                last_val = 1 << j
            end
            y += last_val << shift
        end
    end
    return y, score
end

function action_check(x::UInt64)
    mask = can_shift(x)
    mask |= (0b11 * can_combine_up_down(x)) << 2
    mask |= 0b11 * can_combine_left_right(x)
    return mask
end

function can_combine_up_down(x::UInt64)
    for i in 0:16:48
        last_val = (x >> i) & 0xf
        for j in 4:4:12
            val = (x >> (i+j)) & 0xf
            flag = !iszero(val) && (val == last_val)
            flag && return true
            last_val = val
        end
    end
    return false
end

function can_combine_left_right(x::UInt64)
    for i in 0:4:12
        last_val = (x >> i) & 0xf
        for j in 16:16:64
            val = (x >> (i+j)) & 0xf
            flag = !iszero(val) && (val == last_val)
            flag && return true
            last_val = val
        end
    end
    return false
end

can_shift(x::UInt64) = can_shift(on_off(x))
can_shift(x::UInt16) = (can_shift_up_down(x) << 2) | can_shift_left_right(x)

function on_off(x::UInt64)
    y = zero(UInt16)
    for (i,n) in enumerate(0:4:60)
        temp = (x >> n) & 0xf
        if !iszero(temp)
            y |= one(y) << (i-1)
        end
    end
    return y
end

function can_shift_up_down(x::UInt16)
    flag = 0b00
    for n in 0:4:12
        col = (x >> n) & 0b1111
        up_flag = !(col==0b0000 || col==0b1111 || col==0b0001 || col==0b0011 || col==0b0111)
        down_flag = !(col==0b0000 || col==0b1111 || col==0b1000 || col==0b1100 || col==0b1110)
        flag |= up_flag << 1 | down_flag
    end
    return flag
end

can_shift_left_right(x::UInt16) = can_shift_up_down(bitboard_transpose(x))

function bitboard_transpose(x::UInt16)
    y = zero(x)
    for i in 3:-1:0, j in 12:-4:0
        y = (y<<1) | (x >> (i+j)) & one(x)
    end
    return y
end

function add_square(rng::AbstractRNG, board::UInt64)
    y = on_off(board)
    i = rand(rng, 1:count_zeros(y))
    k = -1
    while i > 0
        k += 1
        if iszero((y >> k)&0b1)
            i -= 1
        end
    end
    val = rand(rng) >= 0.9 ? UInt64(2) : UInt64(1)
    board |= val << (4*k)
    return board
end

function get_rand_valid_action(env::My2048)
    mask = env.valid_action_mask.chunks[]
    N = count_ones(mask)

    if N == 0
        return 0
    elseif N == 1
        return 1 + trailing_zeros(mask)  
    end

    i = 0b1 + (rand(UInt) & 0b11)
    while i > N
        i = 0b1 + (rand(UInt) & 0b11)
    end

    for k = 0:3
        i -= (mask>>k & 0b1)
        if iszero(i)
            return k+1
        end
    end

    return -1
end


end