using BenchmarkTools

board = UInt64

# [
#     1 5 9 13
#     2 6 10 14
#     3 7 11 15
#     4 8 12 16
# ]



function move_up(x::UInt64)
    y = zero(x)
    shift_vec = 0x0010001000100010
    for i in 12:-4:0, j in 0:16:64
        v = (x >> i) & (UInt64(0xf) << j)
        iszero(v) && continue
        shift_vec -= 4 << j
        shift = (shift_vec >> j) & UInt64(0xff)
        y |= v << shift
    end
    return y
end

function move_up(x::UInt64)
    y = zero(x)
    shift_vec = 0x0010001000100010
    for i in 12:-4:0, j in 0:16:64
        v = (x >> i) & (UInt64(0xf) << j)
        iszero(v) && continue
        shift_vec -= 4 << j
        shift = (shift_vec >> j) & UInt64(0xff)
        y |= v << shift
    end
    return y
end

z = 0xf0fab0aa00af0f00
move_up(z) == 0xffa0baa0af00f000
@btime move_up($z)


function move_up_vec(v)
    j = 1
    for i = 1:4
        iszero(v[i]) && continue
        v[j] = v[i]
        j += 1
    end
    v[j:end] .= 0
    return v
end

function move_up_vec(M::Matrix)
    move_up_vec.(eachcol(M))
    M
end

v = rand([0,1],(4,4))
@btime move_up_vec(v)