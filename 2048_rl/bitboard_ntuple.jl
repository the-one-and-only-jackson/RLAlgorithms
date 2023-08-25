include("2048_bitboard.jl")
using CommonRLInterface
using Plots
using Statistics: mean, std
using Flux: softmax, logsumexp, softmax!
using Parameters: @with_kw
using BenchmarkTools
using ProgressMeter
using StaticArrays
using BSON


@with_kw struct NTupleNet{T<:AbstractFloat}
    idx_vec::NTuple{8, UInt64}
    tup_length::Int
    value::Vector{T} = zeros(T, 16^tup_length)
    policy::Matrix{T} = zeros(T, 4, 16^tup_length)
    @assert 0 < tup_length < 7 "Tuple length too large (memory)."
end

Base.eltype(::NTupleNet{T}) where T = T

function NTupleNet(idxs::AbstractMatrix; kwargs...)
    NTupleNet(findall(Bool.(idxs)); sz = size(idxs), kwargs...)
end

function NTupleNet(
    idxs::AbstractVector{CartesianIndex{2}};
    T::Type = Float32,
    sz::Tuple = (4,4)
    )

    tup_length = length(idxs)

    cart_idx_vec = Vector{Vector{CartesianIndex{2}}}()
    for n in 1:4, v in [idxs, mytranspose.(idxs)]
        push!(cart_idx_vec, [myrotate(i, sz, n) for i in v] )
    end

    idx_vec = Tuple(cart_to_lin(idxs,sz) for idxs in cart_idx_vec)

    return NTupleNet{T}(; idx_vec, tup_length)
end

cart_to_lin(idx::CartesianIndex{2}, sz::Tuple{Int,Int}) = idx[1] + (idx[2]-1)*sz[2]

function cart_to_lin(idx_vec::Vector{CartesianIndex{2}}, sz::Tuple{Int,Int})
    key::UInt64 = 0
    for idx in Iterators.reverse(idx_vec)
        i = cart_to_lin(idx,sz) - 1 # zero indexing
        key = key << 4 | i
    end
    return key
end

function myrotate(i, sz, n)
    for _ in 1:n
        i = myrotate(i, sz)
    end
    return i
end
myrotate(i::CartesianIndex{2}, sz::Tuple) = CartesianIndex(i[2], 1 + sz[1] - i[1])

mytranspose(i::CartesianIndex{2}) = CartesianIndex(i[2],i[1])

encode(nt::NTupleNet, state::UInt64) = typeof(nt.idx_vec)(1+encode(state, key, nt.tup_length) for key in nt.idx_vec)

function encode(state::UInt64, key::UInt64, keylength::Integer)
    encoded_state = zero(UInt64)
    for i = 0:keylength-1
        j = 4*i
        square = (key >> j) & UInt64(0xf)
        shift = 4 * square
        val = (state >> shift) & UInt64(0xf)
        encoded_state |= val << j
    end
    return encoded_state
end

@with_kw struct NTuple_array{T<:AbstractFloat}
    NTs::Vector{NTupleNet{T}}
    n::Int = sum(length(nt.idx_vec) for nt in NTs)
end

Base.eltype(::NTuple_array{T}) where T = T

function NTuple_array(idx_vec; kw_args...)
    NTs = [NTupleNet(idxs; kw_args...) for idxs in idx_vec]
    NTuple_array(; NTs)
end

@with_kw mutable struct Buffer{
    VS <: AbstractVector{UInt64}, 
    VT <: AbstractVector{<:AbstractFloat}, 
    MT <: AbstractMatrix{<:AbstractFloat},
    BT <: AbstractVector{Bool},
    LT <: AbstractVector{<:AbstractFloat}
    }

    update_steps::Int
    env::Bitboard2048.My2048 = Bitboard2048.My2048()
    state_vec::VS = VS(undef, update_steps) .= 0
    value_vec::VT = VT(undef,update_steps) .= 0
    reward_vec::VT = VT(undef,update_steps) .= 0
    policy_grad_vec::MT = MT(undef,(4,update_steps)) .= 0
    entropy_vec::VT = VT(undef,update_steps) .= 0
    done_vec::BT = BT(undef,update_steps) .= 0
    advantage_vec::VT = VT(undef,update_steps) .= 0
    idx::Int = 1
    episode_step::Int = 0
    episode_steps::Vector{Int} = Int[]
    episode_scores::Vector{Int} = Int[]
    biggest_tile::Int = 0
    logits::LT = @MVector zeros(eltype(policy_grad_vec), 4)

    @assert eltype(logits) == eltype(policy_grad_vec)
end

@with_kw mutable struct ParBuffer{
    VS <: AbstractArray{UInt64, 2}, 
    VT <: AbstractArray{<:AbstractFloat, 2}, 
    MT <: AbstractArray{<:AbstractFloat, 3},
    BT <: AbstractArray{Bool, 2}
    }

    update_steps::Int
    n_envs::Int
    
    state_vec::VS = VS(undef, update_steps, n_envs) .= 0
    value_vec::VT = VT(undef, update_steps, n_envs) .= 0
    reward_vec::VT = VT(undef, update_steps, n_envs) .= 0
    policy_grad_vec::MT = MT(undef, 4, update_steps, n_envs) .= 0
    entropy_vec::VT = VT(undef, update_steps, n_envs) .= 0
    done_vec::BT = BT(undef, update_steps, n_envs) .= 0
    advantage_vec::VT = VT(undef, update_steps, n_envs) .= 0
    episode_steps::Vector{Int} = Int[]
    episode_scores::Vector{Int} = Int[]
    biggest_tile::Int = 0

    buff_vec::Vector{<:Buffer} = [
        Buffer(;
            update_steps,
            state_vec = view(state_vec, :, i),
            value_vec = view(value_vec, :, i),
            reward_vec = view(reward_vec, :, i),
            policy_grad_vec = view(policy_grad_vec, :, :, i),
            entropy_vec = view(entropy_vec, :, i),
            done_vec = view(done_vec, :, i),
            advantage_vec = view(advantage_vec, :, i)
        ) 
        for i in 1:n_envs
    ]
end

ParBuffer(T,A; kw_args...) = ParBuffer{A{UInt64,2}, A{T,2}, A{T,3}, A{Bool,2}}(; kw_args...)

Buffer(T; kw_args...) = Buffer(T,Array; kw_args...)
Buffer(T,A; kw_args...) = Buffer{T, A{UInt64,1}, A{T,1}, A{T,2}, A{Bool,1}}(; kw_args...)

get_value(nt_arr::NTuple_array, state) = sum(nt->get_value(nt, state), nt_arr.NTs)
get_value(nt::NTupleNet, state) = sum(key->nt.value[key], encode(nt, state))

function get_action(
    nt_arr::NTuple_array{T}, 
    state::UInt64, 
    action_mask::BitVector, 
    logits::AbstractVector{T}=MVector{4,T}(0,0,0,0)
    ) where T

    # probs is mutated twice, name changed for clarity
    logits = mask_logits(get_logits!(logits, nt_arr, state), action_mask)
    probs = my_softmax!(logits)
    entropy = -sum(x->ifelse(iszero(x),zero(x),x*log(x)), probs)
    action = sample_discrete(probs, action_mask)
    policy_grad = calc_policy_grad!(probs, action)
    return action, policy_grad, entropy
end

function get_logits!(logits::AbstractVector{T}, nt_arr::NTuple_array{T}, state) where T
    for nt in nt_arr.NTs, key in encode(nt, state), i=1:4
        @inbounds logits[i] += nt.policy[i, key]
    end
    return logits
end

function mask_logits(logits::AbstractVector{T}, mask::AbstractVector{Bool}) where T
    for (i, bit) in enumerate(mask)
        if bit == false
            logits[i] = typemin(T)
        end
    end
    return logits
end

function my_softmax!(x::AbstractVector{T}) where {T<:AbstractFloat}
    max_ = @fastmath reduce(max, x; init = typemin(T))
    if isfinite(max_)
        @fastmath @. x = exp(x - max_)
    else
        @fastmath @. x = ifelse(isequal(max_,Inf), ifelse(isequal(x,Inf), 1, 0), exp(x - max_))
    end
    x ./= sum(x)
end

function calc_policy_grad!(probs::AbstractVector{<:AbstractFloat}, action::Integer)
    policy_grad = probs # mutating
    policy_grad .*= -1
    policy_grad[action] += 1
    return policy_grad
end

sample_discrete(probs, mask::BitVector) = sample_discrete(probs, mask.chunks[])
function sample_discrete(probs::AbstractVector{<:AbstractFloat}, mask::UInt64)
    N = count_ones(mask)

    N == 0 && return 0 # 1 indexing, indicates no valid action
    N == 1 && return 1 + trailing_zeros(mask)

    action = sample_discrete(probs)

    if ((mask >> (action-1)) & 0b1) == 0b1
        return action
    end

    i = 0b1 + (rand(UInt8) & 0b11)
    while i > N
        i = 0b1 + (rand(UInt8) & 0b11)
    end

    for k = 0:3
        i -= (mask>>k & 0b1)
        iszero(i) && return k+1
    end

    return 0
end

function sample_discrete(v::AbstractVector{T}) where T
    t = rand(T) * sum(v)
    cw = zero(T)
    for (i, w) in enumerate(v) 
        (cw += w) >= t && return i
    end
    return length(v)
end

@with_kw mutable struct LongStats{X<:AbstractFloat,Y<:AbstractFloat}
    x::Vector{X} = X[]
    y::Vector{Y} = Y[]
    x_buffer::Vector{X} = X[]
    y_buffer::Vector{Y} = Y[]
    buff_len::Int = 1
end

LongStats() = LongStats{Float32,Float32}()

add_stat!(LS::LongStats{X,Y}, x, y) where {X,Y} = add_stat!(LS, X(x), Y(y))
function add_stat!(LS::LongStats{X,Y}, x::X, y::Y) where {X,Y}
    push!(LS.x_buffer, x)
    push!(LS.y_buffer, y)
    manage_buffer!(LS)
    manage_data!(LS)
    nothing
end

function manage_buffer!(LS::LongStats)
    if length(LS.x_buffer) == LS.buff_len
        push!(LS.x, mean(LS.x_buffer))
        push!(LS.y, mean(LS.y_buffer))
        empty!(LS.x_buffer)
        empty!(LS.y_buffer)
    end
    nothing
end

function manage_data!(LS::LongStats)    
    if length(LS.x) >= 5_000
        for i = 1:500
            idxs = (-9:0) .+ (10*i)
            LS.x[i] = mean(LS.x[idxs])
            LS.y[i] = mean(LS.y[idxs])
        end
        deleteat!(LS.x, 501:5_000)
        deleteat!(LS.y, 501:5_000)
        LS.buff_len *= 10
    end
    nothing
end

get_data(LS::LongStats) = (x=copy(LS.x), y=copy(LS.y))

function smooth_stats(v::Vector{T}; k=10) where T
    N = length(v)
    μ = similar(v)
    for i = 1:N
        delta_l = min(k, i-1)
        delta_r = min(k, N-i)
        idxs = (i-delta_l):(i+delta_r)
        μ[i] = mean(view(v, idxs))
    end
    return μ
end

function take_step!(buffer::Buffer, nt_arr::NTuple_array)
    state = observe(buffer.env)
    mask = valid_action_mask(buffer.env)
    action, policy_grad, entropy = get_action(nt_arr, state, mask, buffer.logits)
    @assert isfinite(entropy)
    reward = act!(buffer.env, action)
    done = terminated(buffer.env)

    buffer.state_vec[buffer.idx] = state
    buffer.reward_vec[buffer.idx] = reward
    buffer.policy_grad_vec[:,buffer.idx] = policy_grad
    buffer.entropy_vec[buffer.idx] = entropy
    buffer.done_vec[buffer.idx] = done

    buffer.episode_step += 1
    buffer.idx = mod1(1+buffer.idx, buffer.update_steps)

    if done
        push!(buffer.episode_steps, buffer.episode_step)
        buffer.episode_step = 0
        push!(buffer.episode_scores, buffer.env.score)
        buffer.biggest_tile = max(buffer.biggest_tile, buffer.env.biggest_tile)

        reset!(buffer.env)
    end

    nothing
end

#=
function get_value_advantage!(buffer::Buffer, nt_arr::NTuple_array{T}; lambda::T) where T
    Threads.@threads for i in 1:buffer.update_steps
        buffer.value_vec[i] = get_value(nt_arr,buffer.state_vec[i])
    end

    last_val = get_value(nt_arr, observe(buffer.env))
    gae!(buffer, last_val; lambda)
end
=#

function get_value_advantage!(parbuf::ParBuffer, nt_arr::NTuple_array{T}; lambda::T) where T
    Threads.@threads for i in 1:length(parbuf.value_vec)
        parbuf.value_vec[i] = get_value(nt_arr, parbuf.state_vec[i])
    end

    Threads.@threads for buffer in parbuf.buff_vec
        last_val = get_value(nt_arr, observe(buffer.env))
        gae!(buffer, last_val; lambda)
    end

    nothing
end

gae!(buffer::Buffer, last_val; lambda) = gae!(buffer.advantage_vec, buffer.reward_vec, buffer.value_vec, buffer.done_vec, last_val; lambda)

function gae!(
    advantages::V, rewards::V, values::V, done_vec::AbstractVector{Bool}, last_val::T; 
    lambda::T=T(0.95)
    ) where {T, V<:AbstractVector{T}}

    advantages[end] = rewards[end] + !done_vec[end]*last_val - values[end]
    for ii in length(advantages)-1:-1:1
        td = rewards[ii] + !done_vec[ii] * values[ii+1] - values[ii]
        advantages[ii] = td + lambda * !done_vec[ii] * advantages[ii+1]
    end

    nothing
end

function update_nt_parallel!(
    nt_arr::NTuple_array{T}, buffer::ParBuffer; 
    value_lr::T, policy_lr::T
    ) where T

    state_vec = reshape(buffer.state_vec, :)
    advantage_vec = reshape(buffer.advantage_vec, :)
    policy_grad_vec = reshape(buffer.policy_grad_vec, 4, :)

    adv_μ = mean(advantage_vec)
    adv_σ = std(advantage_vec; mean=adv_μ)
    @. policy_grad_vec *= (policy_lr/adv_σ) * (advantage_vec' - adv_μ)

    advantage_vec .*= value_lr

    val_task = map(nt_arr.NTs) do nt
        Threads.@spawn _update_value!(nt, state_vec, advantage_vec)
    end

    pol_task = map(nt_arr.NTs) do nt
        Threads.@spawn _update_policy!(nt, state_vec, policy_grad_vec)
    end

    wait.(val_task)
    wait.(pol_task)

    nothing
end

function _update_value!(nt::NTupleNet, state_vec::AbstractVector, value_grad_vec::AbstractVector)
    for (state, value_grad) in zip(state_vec, value_grad_vec), key in encode(nt, state)
        nt.value[key] += value_grad
    end
    nothing
end

function _update_policy!(nt::NTupleNet, state_vec::AbstractVector, policy_grad_mat::AbstractMatrix)
    for (state, policy_grad) in zip(state_vec, eachcol(policy_grad_mat)), key in encode(nt, state)
        view(nt.policy,:,key) .+= policy_grad 
    end
    nothing
end

function collect_steps!(parbuf::ParBuffer, nt_arr::NTuple_array)
    Threads.@threads for buffer in parbuf.buff_vec
        collect_steps!(buffer, nt_arr)
    end
    for buffer in parbuf.buff_vec
        append!(parbuf.episode_steps, buffer.episode_steps)
        empty!(buffer.episode_steps)
        append!(parbuf.episode_scores, buffer.episode_scores)
        empty!(buffer.episode_scores)
    end
    parbuf.biggest_tile = maximum(buffer.biggest_tile for buffer in parbuf.buff_vec)
    nothing
end

function collect_steps!(buffer::Buffer, nt_arr::NTuple_array)
    for _ in 1:buffer.update_steps
        take_step!(buffer, nt_arr)
    end
    nothing
end

function experiment(nt_arr::NTuple_array{T};
    file = "2048_curve.png", 
    n_steps = 1_000_000,
    update_steps = 1_000,
    n_envs = 10,
    value_lr = 0.1/(update_steps*n_envs),
    policy_lr = 0.01,
    lambda = 0.95,
    plot_dt = 1.0
    ) where T

    (value_lr, policy_lr, lambda) = T.((value_lr, policy_lr, lambda))

    buffer = ParBuffer(T, Array; update_steps, n_envs)

    mean_episode_length = LongStats()
    mean_score = LongStats()
    mean_entropy = LongStats()
    biggest_tile_hist = (x=Int[], y=Int[])

    info = [mean_episode_length, mean_score, mean_entropy, biggest_tile_hist]

    last_plot_time = time()
    last_save_time = time()

    steps_per_update = Int(update_steps*n_envs)
    @showprogress for global_step in steps_per_update:steps_per_update:n_steps
        collect_steps!(buffer, nt_arr)
        get_value_advantage!(buffer, nt_arr; lambda)
        update_nt_parallel!(nt_arr, buffer; value_lr, policy_lr)

        for (dst,src) in zip((mean_episode_length, mean_score), (buffer.episode_steps, buffer.episode_scores))
            foreach(x->add_stat!(dst, global_step, x), src)
            empty!(src)    
        end

        add_stat!(mean_entropy, global_step, mean(buffer.entropy_vec))

        if isempty(biggest_tile_hist.y) || buffer.biggest_tile > biggest_tile_hist.y[end]
            append!(biggest_tile_hist.x, fill(global_step,2))
            append!(biggest_tile_hist.y, fill(buffer.biggest_tile,2))
        end

        if time()-last_plot_time > plot_dt   
            last_plot_time = time()
            biggest_tile_hist.x[end] = global_step
            plotting_fun(file, mean_episode_length, mean_score, biggest_tile_hist, mean_entropy)
        end

        if time() - last_save_time > 60*30
            last_save_time = time()
            BSON.@save "8_21_v1" nt_arr info
        end
    end

    return nt_arr, info
end

# NOTE: plots are bad due to how LongStats smooths linearly 
function plotting_fun(file, mean_episode_length, mean_score, biggest_tile_hist, mean_entropy)
    (x,y) = biggest_tile_hist
    common = (; label=false, xlimits=(x[1],x[end]))
    p4 = plot(x, y; ylabel="Max Tile", common...)

    common_logy = common
    (x,y) = get_data(mean_episode_length)
    p1 = plot(x,smooth_stats(y); ylabel="Episode Length", common_logy...)

    (x,y) = get_data(mean_score)
    p2 = plot(x,smooth_stats(y); ylabel="Score", common_logy...)

    (x,y) = get_data(mean_entropy)
    p3 = plot(x,smooth_stats(y); ylabel="Entropy", common_logy...)

    plt = plot(p1,p2,p3,p4; layout=(2,2), size=(900,600))

    savefig(plt,file)

    nothing
end

nt_matrices = [
    [
        1 1 0 0;
        1 1 0 0;
        1 1 0 0;
        0 0 0 0
    ],
    [
        0 1 1 0;
        0 1 1 0;
        0 1 1 0;
        0 0 0 0
    ]
]

nt_arr = experiment(NTuple_array(nt_matrices; T=Float16, sz = (4,4)); 
    file= "2048_curve_11.png", 
    n_steps = 10_000_000,
    update_steps = 10,
    n_envs = 10,
    policy_lr = 1e-1,
    value_lr = 1e-6,
    plot_dt = 0.5,
    lambda = 0.99
)

nt_arr.NTs[1].policy



experiment(NTuple_array(nt_matrices; T=Float16, sz = (4,4)); 
    file= "2048_curve_12.png", 
    n_steps = 100_000_000,
    update_steps = 100,
    n_envs = 100,
    policy_lr = 0.001,
    plot_dt = 0.5,
    lambda = 0.99
)

nt_matrices = [
    [
        1 1 1 1;
        1 1 0 0;
        0 0 0 0;
        0 0 0 0
    ],
    [
        0 0 0 0;
        1 1 1 1;
        1 1 0 0;
        0 0 0 0
    ],
    [
        1 1 1 0;
        1 1 1 0;
        0 0 0 0;
        0 0 0 0
    ],
    [
        0 0 0 0;
        1 1 1 0;
        1 1 1 0;
        0 0 0 0
    ]
]

nt_arr, info = experiment(NTuple_array(nt_matrices; T=Float32, sz = (4,4)); 
    file= "2048_curve_4.png", 
    n_steps = 25_000_000_000,
    update_steps = 100,
    n_envs = 100,
    policy_lr = 0.005,
    plot_dt = 0.5,
    lambda = 0.99
)





function check_2048(env::Bitboard2048.My2048)
    board = env.board
    for i in 0:4:60
        val = (board >> i) & 0xf
        if val >= 0xb
            return true
        end
    end
    return false
end

using BSON
BSON.@save "10_10_v1" nt

function evaluate(nt, env::Bitboard2048.My2048)
    reset!(env)
    while !terminated(env) && !check_2048(env)
        state = observe(env)
        mask = valid_action_mask(env)
        a, _ = get_action(nt, state, mask)
        act!(env, a)
    end
    return check_2048(env)
end

[evaluate(nt, env) for _ in 1:1000] |> mean




# first work on speeding up / parallizing
# then try adjusting the encoding [a,3,6,0] -> -> nonzero -> minus(min)+1 -> [8,1,4,0]
# might not work




d = rand(1:16^6, 100_000)
v = rand(Float16, 100_000)
lut = [Threads.Atomic{Float16}(0) for _ in 1:16^6]


@btime grad_table($lut, $d, $v)

@btime fill!($lut, NaN)
@btime parfill!($lut, NaN16, 12)
@btime parfill!($lut, NaN16, 48)

function parfill!(v::AbstractArray{T}, x::T, n_threads=Threads.nthreads()) where T
    chunks = Iterators.partition(v, length(v) ÷ n_threads)
    tasks = map(chunks) do chunk
        Threads.@spawn fill!(chunk, x)
    end
    wait.(tasks)
    nothing
end

function grad_table(lut::AbstractVector{Threads.Atomic{T}}, idx::AbstractVector{Int}, val::AbstractVector{T}, n_threads=Threads.nthreads()) where T
    itr = zip(idx, val)
    chunks = Iterators.partition(itr, length(itr) ÷ n_threads)
    tasks = map(chunks) do chunk
        Threads.@spawn foreach((i,v)->Threads.atomic_add!(lut[i], v), chunk[1], chunk[2]) 
    end
    wait.(tasks)
    return lut
end


itr = zip(d,v)
chunks = Iterators.partition(itr, length(itr) ÷ 24)
tasks = map(chunks) do chunk
    Threads.@spawn foreach((i,v)->(lut[i] += v), chunk[1], chunk[2]) 
end

