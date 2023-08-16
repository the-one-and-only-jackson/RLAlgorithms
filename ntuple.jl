using Random: randperm

include("2048_env.jl")
using CommonRLInterface

using Plots

using Statistics: mean, std
using Flux: softmax, logsumexp

# add N and M as paramters -> n_tuple instead of n_tuple_4 for 0x0:0xf
# Mapping from values to indices ? 
struct NTuple{T<:AbstractFloat}
    idx_vec::AbstractVector{AbstractVector{CartesianIndex{2}}}
    value::AbstractVector{T}
    policy::AbstractMatrix{T}
    keyval::Function # maxtrix entry -> integer in [0, n_vals-1]
    n_vals::Integer
end

Base.eltype(::NTuple{T}) where T = T

function NTuple(idxs::AbstractMatrix; kwargs...)
    NTuple(findall(Bool.(idxs)); sz = size(idxs), kwargs...)
end

function NTuple(
    idxs::AbstractVector{CartesianIndex{2}};
    n_vals::Integer,
    keyval::Function = identity,
    T::Type = Float32,
    value_init::Function = zeros,
    sz::Tuple = (4,4)
    )

    idx_vec = Vector{Vector{CartesianIndex{2}}}()
    for n in 1:4, v in [idxs, mytranspose.(idxs)]
        push!(idx_vec, [myrotate(i, sz, n) for i in v] )
    end
    unique!(idx_vec)


    value = value_init(T, n_vals^length(idxs))
    policy = zeros(T, 4, n_vals^length(idxs))

    return NTuple{T}(idx_vec, value, policy, keyval, n_vals)
end

function myrotate(i, sz, n)
    for _ in 1:n
        i = myrotate(i, sz)
    end
    return i
end
myrotate(i::CartesianIndex{2}, sz::Tuple) = CartesianIndex(i[2], 1 + sz[1] - i[1])

mytranspose(i::CartesianIndex{2}) = CartesianIndex(i[2],i[1])

function encode(nt::NTuple, M::AbstractMatrix)
    encoded_state = ones(Int, length(nt.idx_vec)) # one indexing
    for (ii,idxs) in enumerate(nt.idx_vec), (jj,idx) in enumerate(idxs)
        digit = nt.keyval(M[idx])
        @assert 0 ≤ digit ≤ nt.n_vals-1
        multiplier = nt.n_vals ^ (jj-1)
        encoded_state[ii] += digit * multiplier
    end
    return encoded_state
end

struct NTuple_array{T<:AbstractFloat}
    NTs::AbstractVector{NTuple{T}}
    n::Integer
end

Base.eltype(::NTuple_array{T}) where T = T

NTuple_array(NTs::AbstractVector{NTuple{T}}, n) where T = NTuple_array{T}(NTs, n)

function NTuple_array(idx_vec; kw_args...)
    NTs = [NTuple(idxs; kw_args...) for idxs in idx_vec]
    n = sum(length(nt.idx_vec) for nt in NTs)
    NTuple_array(NTs, n)
end

function get_value(nt::NTuple, M::AbstractMatrix)
    return sum(nt.value[encode(nt,M)])
end

function get_value(nt_arr::NTuple_array, M::AbstractMatrix)
    sum(get_value(nt,M) for nt in nt_arr.NTs)
end

function policy_grad!(
    nt_arr::NTuple_array{T}, 
    state::AbstractMatrix, 
    value_grad::T,
    policy_grad::AbstractVector{T};
    value_lr::T = T(0.01),
    policy_lr::T = T(0.01)
    ) where T

    value_lr /= nt_arr.n
    policy_lr /= nt_arr.n

    for nt in nt_arr.NTs, key in encode(nt, state)
        nt.value[key] += value_lr * value_grad
        nt.policy[:, key] .+= policy_lr * policy_grad
    end

    nothing
end

function my_log2(x::Unsigned)
    iszero(x) && return 0
    for k=0:Sys.WORD_SIZE-1
        check_val = one(x) << k
        x == check_val && return k
    end
    @assert false "Not a power of 2 (or zero)"
end

function policy(
    nt_arr::NTuple_array{T}, 
    state::AbstractMatrix, 
    action_mask::AbstractVector{Bool};
    big_num::T = T(1e8)
    ) where T

    logits = zeros(T, 4)
    for nt in nt_arr.NTs, key in encode(nt, state)
        logits .+= nt.policy[:, key]
    end
    logits .-= big_num * .!action_mask

    log_probs = logits .- logsumexp(logits)
    probs = softmax(log_probs)
    action = sample_discrete(probs)
    entropy = - (log_probs' * probs)

    grad_policy = -probs[action] * probs
    grad_policy[action] += probs[action]
    grad_policy ./= probs[action]

    return action, grad_policy, entropy
end

function sample_discrete(v::AbstractVector{T}) where T
    t = rand(T) * sum(v)
    cw = zero(T)
    for (i, w) in enumerate(v) 
        (cw += w) >= t && return i
    end
    return length(wv)
end

function run_ep_sarsa(env, nt_arr::NTuple_array{T}; lambda::T = T(0.95)) where {T}
    info = Dict(:step=>0.0)

    state_vec = typeof(observe(env))[]
    policy_grad_vec = Vector{T}[]
    rew_vec = T[]
    val_vec = T[]
    entropy_vec = T[]

    reset!(env)
    while !terminated(env)
        state = observe(env)
        action, policy_grad, entropy = policy(nt_arr, state, valid_action_mask(env))
        val = get_value(nt_arr, state)
        # reward = act!(env, action) |> T |> x->max(zero(T), log2(x))
        reward = act!(env, action) |> T

        push!(state_vec, state)
        push!(policy_grad_vec, policy_grad)
        push!(rew_vec, reward)
        push!(val_vec, val)
        push!(entropy_vec, entropy)
        info[:step] += 1
    end

    td_error_vec = lambda_return(rew_vec; lambda) .- val_vec

    for (state, value_grad, policy_grad) in zip(state_vec, td_error_vec, policy_grad_vec)
        policy_grad!(nt_arr, state, value_grad, policy_grad)
    end

    info[:score] = env.score
    info[:td_error] = mean(abs, td_error_vec)
    info[:entropy] = mean(entropy_vec)

    return info
end

function vanilla_policy_grad(env::AbstractEnv, nt_arr::NTuple_array{T}) where T
    info_out = Dict()

    state_vec = []
    td_error_vec = []
    policy_grad_vec = []

    for _ in 1:100
        info = run_ep(env, nt_arr)
        score = env.score
        advantage = lambda_return(info[:reward]; lambda) .- info[:value]
    end
end

function run_ep(env::AbstractEnv, nt_arr::NTuple_array{T}) where {T}
    info = (
        :state=>typeof(observe(env))[],
        :policy_grad=>T[],
        :entropy=>T[],
        :reward=>T[],
        :value=>T[]
    )

    reset!(env)
    while !terminated(env)
        state = observe(env)
        action, policy_grad, entropy = policy(nt_arr, state, valid_action_mask(env))
        value = get_value(nt_arr, state)
        # reward = act!(env, action) |> T |> x->max(zero(T), log2(x))
        reward = act!(env, action) |> T

        push!(info[:state], state)
        push!(info[:policy_grad], policy_grad)
        push!(info[:reward], reward)
        push!(info[:value], value)
        push!(info[:entropy], entropy)
    end

    return info
end

function lambda_return(rewards::Vector{T}; gamma=one(T), lambda=one(T)) where T
    c = T(gamma * lambda)
    returns = copy(rewards)
    for i = length(returns)-1:-1:1
        returns[i] += c * returns[i+1]
    end
    return returns
end

function plot_fun(info; c=1)
    p_vec = [plot(smooth_stats(val)...; c, label=false, ylabel=key) for (key,val) in info]
    plot(p_vec..., layout=(2,2))
end

function smooth_stats(vin::Vector{T}; k=10) where T
    x, v = compress_vec(vin)
    N = length(v)
    μ = similar(v)
    for i = 1:N
        delta_l = min(k, i-1)
        delta_r = min(k, N-i)
        idxs = (i-delta_l):(i+delta_r)
        μ[i] = mean(view(v, idxs))
    end
    return x, μ
end
function compress_vec(v::Vector{T}; max_n=1_000) where T
    T2 = float(T)
    N = length(v)
    k = Int(ceil(N/max_n))
    if k <= 1
        v2 = Vector{T2}(v)
    else
        N2 = Int(ceil(N/k))
        v2 = Vector{T2}(undef, N2)
        for i = 1:N2
            i1 = 1 + k*(i-1)
            i2 = min(k*i, N)
            v2[i] = mean(view(v,i1:i2))
        end
    end
    x = (k+1)/2.0 .+ (k * (0:length(v2)-1))
    return x, v2
end

function experiment(; 
    file= "2048_curve.png", 
    n_steps = 100_000,
    T = Float16,
    v_init = 0
    )

    nt_arr = NTuple_array(
        [
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
            ],
            [
                0 0 0 1;
                0 0 0 1;
                0 0 0 1;
                0 0 0 1
            ],
            [
                0 0 1 0;
                0 0 1 0;
                0 0 1 0;
                0 0 1 0
            ]
        ];
        n_vals = 16,
        keyval = my_log2,
        T,
        sz = (4,4)
    )

    for nt in nt_arr.NTs
        fill!(nt.value, T(v_init))
    end

    env = GameEnv.My2048()

    info = Dict{Symbol, Vector}()

    for t in 1:n_steps        
        for (key, val) in run_ep_sarsa(env, nt_arr)
            push!(get!(info, key, typeof(val)[]), val)
        end

        if t % 100 == 0
            plt = plot_fun(info)
            savefig(plt,file)
        end
    end
    return nt_arr, env
end

experiment(; 
    file= "2048_curve_1.png", 
    n_steps = 10_000,
    T=Float32
)



# function baseline(env)
#     step = 0
#     reset!(env)
#     while !terminated(env)
#         step += 1
#         a = rand(findall(valid_action_mask(env)))
#         act!(env, a)
#     end
#     return env.score, step
# end

# score, step = UInt16[], Int[]
# for _ in 1:1000
#     push!.((score, step), GameEnv.My2048() |> baseline)
# end
# (score, step) .|> mean # 1088, 118

# n tuple policy? 
 
# struct moving_avg_filter
#     y::Vector{Float64}
#     data::Vector{Float64}
#     k::Int
# end
# moving_avg_filter(k) = moving_avg_filter(Float64[], Float64[], k)

# function add_val(filter::moving_avg_filter, x)
#     y, data, k = filter.y, filter.data, filter.k
#     push!(data, x)
#     L = length(data)
#     if L > k
#         push!(y, mean(data[max(1,L-2*k):L]))
#     end
#     nothing
# end


# actor
# state -> (4 x m values) -> sum -> 4 values -> softmax -> rand





