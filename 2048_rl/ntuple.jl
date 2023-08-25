
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

function my_log2(x::Unsigned)
    iszero(x) && return 0
    for k=0:Sys.WORD_SIZE-1
        check_val = one(x) << k
        x == check_val && return k
    end
    @assert false "Not a power of 2 (or zero)"
end

function update_nt_arr!(
    nt_arr::NTuple_array{T}, 
    states::AbstractVector{<:AbstractMatrix}, 
    advantages::AbstractVector{T}, 
    policy_grads::AbstractMatrix{T}; 
    value_lr::T = T(0.1), 
    policy_lr::T = T(0.1)
    ) where T

    value_lr /= nt_arr.n
    policy_lr /= nt_arr.n

    adv_μ = mean(advantages)
    adv_σ = std(advantages)

    for (state, advantage, policy_grad) in zip(states, advantages, eachcol(policy_grads))
        for nt in nt_arr.NTs, key in encode(nt, state)
            nt.value[key] += value_lr * advantage
            nt.policy[:, key] .+= policy_lr * policy_grad * (advantage - adv_μ)/adv_σ
        end
    end

    nothing
end

function get_actionvalue(
    nt_arr::NTuple_array{T}, 
    state::AbstractMatrix, 
    action_mask::AbstractVector{Bool};
    big_num::T = T(1e8)
    ) where T

    logits = zeros(T, 4)
    value = zero(T)
    for nt in nt_arr.NTs, key in encode(nt, state)
        logits .+= nt.policy[:, key]
        value += nt.value[key]
    end
    logits .-= big_num * .!action_mask

    log_probs = logits .- logsumexp(logits)
    probs = softmax(log_probs)
    action = sample_discrete(probs)
    entropy = - (log_probs' * probs)

    policy_grad = -probs[action] * probs
    policy_grad[action] += probs[action]
    policy_grad ./= probs[action]    

    return action, policy_grad, entropy, value
end

function sample_discrete(v::AbstractVector{T}) where T
    t = rand(T) * sum(v)
    cw = zero(T)
    for (i, w) in enumerate(v) 
        (cw += w) >= t && return i
    end
    return length(v)
end


function gae!(advantages::AbstractVector{T}, rewards::AbstractVector{T}, values::AbstractVector{T}, done_vec::AbstractVector{Bool}, last_val::T; lambda::T=T(0.95)) where T
    advantages[end] = rewards[end] + !done_vec[end]*last_val - values[end]
    for ii in length(advantages)-1:-1:1
        advantages[ii] = rewards[ii] + !done_vec[ii] * values[ii+1] - values[ii] + lambda * !done_vec[ii] * advantages[ii+1]
    end
    return advantages
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

function experiment(nt_arr::NTuple_array{T}; 
    file= "2048_curve.png", 
    n_eps = 100_000,
    update_steps = 1_000,
    value_lr::T = T(0.01),
    policy_lr::T = T(0.01),
    lambda::T = T(0.95)
    ) where T

    env = GameEnv.My2048()
    reset!(env)

    info = Dict{Symbol, Vector}(:step=>Int[], :entropy=>T[], :score=>typeof(env.score)[])

    state_vec = Vector{typeof(observe(env))}(undef, update_steps)
    value_vec = Vector{T}(undef,update_steps)
    reward_vec = Vector{T}(undef,update_steps)
    policy_grad_vec = Matrix{T}(undef,(4,update_steps))
    done_vec = Vector{Bool}(undef,update_steps)
    advantage_vec = Vector{T}(undef,update_steps)

    ep_step::Int = 0
    ep_entropy::T = 0

    ep_count = 0
    plot_flag = false

    while ep_count < n_eps
        for step in 1:update_steps
            state_vec[step] = observe(env)
            action, policy_grad_vec[:,step], entropy, value_vec[step] = get_actionvalue(nt_arr, state_vec[step], valid_action_mask(env))
            reward_vec[step] = act!(env, action) |> T # |> x->max(zero(T), log2(x))
            done_vec[step] = terminated(env)

            ep_step += 1
            ep_entropy += entropy

            if done_vec[step]
                push!(info[:step], ep_step)
                push!(info[:entropy], ep_entropy/ep_step)
                push!(info[:score], env.score)
                reset!(env)
                ep_step = 0
                ep_entropy = 0
                ep_count += 1
                plot_flag |= (ep_count % 50 == 0)
            end
        end

        _, _, _, last_val = get_actionvalue(nt_arr, observe(env), valid_action_mask(env))
        gae!(advantage_vec, reward_vec, value_vec, done_vec, last_val; lambda)

        update_nt_arr!(nt_arr, state_vec, advantage_vec, policy_grad_vec; value_lr, policy_lr)

        if plot_flag
            println(ep_count)
            plt = plot_fun(info)
            savefig(plt,file)
            plot_flag = false
        end
    end
    return nt_arr, env
end

nt_arr = NTuple_array(
    [
        [
            1 0 0 0;
            1 0 0 0;
            1 0 0 0;
            1 0 0 0
        ],
        [
            0 1 0 0;
            0 1 0 0;
            0 1 0 0;
            0 1 0 0
        ],
        [
            1 1 0 0;
            1 1 0 0;
            0 0 0 0;
            0 0 0 0
        ],
        [
            0 1 1 0;
            0 1 1 0;
            0 0 0 0;
            0 0 0 0
        ],
        [
            0 0 0 0;
            0 1 1 0;
            0 1 1 0;
            0 0 0 0
        ]
    ];
    n_vals = 16,
    keyval = my_log2,
    T = Float32,
    sz = (4,4)
)

experiment(nt_arr; 
    file= "2048_curve_1.png", 
    n_eps = 1_000_000,
    value_lr = Float32(0.01),
    policy_lr = Float32(0.1)
)

nt_arr = NTuple_array(
    [
        [
            1 0 0 0;
            1 0 0 0;
            1 0 0 0;
            1 0 0 0
        ],
        [
            0 1 0 0;
            0 1 0 0;
            0 1 0 0;
            0 1 0 0
        ],
        [
            1 1 0 0;
            1 1 0 0;
            0 0 0 0;
            0 0 0 0
        ],
        [
            0 1 1 0;
            0 1 1 0;
            0 0 0 0;
            0 0 0 0
        ],
        [
            0 0 0 0;
            0 1 1 0;
            0 1 1 0;
            0 0 0 0
        ]
    ];
    n_vals = 16,
    keyval = my_log2,
    T=Float32,
    sz = (4,4)
)

experiment(nt_arr; 
    file= "2048_curve_2.png", 
    n_eps = 10_000,
    value_lr = Float32(0.001),
    policy_lr = Float32(0.1)
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
    T = Float32,
    sz = (4,4)
)

experiment(nt_arr; 
    file= "2048_curve_3.png", 
    n_eps = 1_000_000,
    value_lr = Float32(0.01),
    policy_lr = Float32(0.1)
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
    T = Float32,
    sz = (4,4)
)

experiment(nt_arr; 
    file= "2048_curve_4.png", 
    n_eps = 10_000,
    value_lr = Float32(0.001),
    policy_lr = Float32(0.1)
)

nt_arr = NTuple_array(
    [
        [
            1 0 0 0;
            1 0 0 0;
            1 0 0 0;
            0 0 0 0
        ],
        [
            0 1 0 0;
            0 1 0 0;
            0 1 0 0;
            0 0 0 0
        ]
    ];
    n_vals = 16,
    keyval = my_log2,
    T = Float32,
    sz = (4,4)
)

experiment(nt_arr; 
    file= "2048_curve_5.png", 
    n_eps = 10_000,
    value_lr = Float32(0.01),
    policy_lr = Float32(0.1)
)

nt_arr = NTuple_array(
    [
        [
            1 0 0 0;
            1 0 0 0;
            1 0 0 0;
            0 0 0 0
        ],
        [
            0 1 0 0;
            0 1 0 0;
            0 1 0 0;
            0 0 0 0
        ]
    ];
    n_vals = 16,
    keyval = my_log2,
    T = Float32,
    sz = (4,4)
)

experiment(nt_arr; 
    file= "2048_curve_6.png", 
    n_eps = 10_000,
    value_lr = Float32(0.01),
    policy_lr = Float32(0.01)
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





# test train value on random policy
# look at advantage

