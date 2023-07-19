module AST

using CommonRLInterface
using Parameters: @with_kw
using Distributions: loglikelihood, Normal, MvNormal
using Statistics: mean, std, cov

using RLAlgorithms.Spaces: Box, product

export AST_distributional

const RL = CommonRLInterface

@with_kw struct AST_distributional <: AbstractEnv
    env::AbstractEnv
    n_steps::Int64
    terminal_cost::Float64 = 5*n_steps*loglikelihood(actions(env).d, mean(actions(env).d))
    info::Dict = Dict(
        :steps=>Int64[],
        :likelihood=>Float64[],
        :KL=>Float64[],
        :fail=>Bool[]
    )
end

function RL.reset!(e::AST_distributional)
    if !isempty(e.info[:steps])
        dist = actions(e.env).d
        max_likelihood = loglikelihood(dist, mean(dist))
        e.info[:likelihood][end] += (e.n_steps - e.info[:steps][end]) * max_likelihood
    end

    for vec in values(e.info)
        push!(vec, zero(eltype(vec)))
    end

    reset!(e.env)
    nothing
end

function RL.act!(e::AST_distributional, a)
    model = actions(e.env).d
    dist = unflatten(model, a)

    x = rand(dist)
    act!(e.env, x)

    likelihood = loglikelihood(model, x)

    # logging
    e.info[:steps][end] += 1
    e.info[:likelihood][end] += likelihood
    e.info[:KL][end] += kl_divergence(dist, model)
    e.info[:fail][end] |= terminated(e.env)

    r = likelihood - e.terminal_cost * time_limit(e)

    return r
end

function unflatten(model::Normal, a)
    μ = mean(model) + std(model) * a[1]
    σ = std(model) * exp(a[2] / 2)
    return Normal(μ, σ)
end

function unflatten(model::MvNormal, a)
    N = length(model)
    μ = mean(model) + sqrt(cov(model)) * a[1:N]
    σ = sqrt(cov(model)) * exp.(a[N+1:end] / 2)
    return MvNormal(μ, σ)
end

function kl_divergence(p::Normal, q::Normal)
    return log(q.σ/p.σ) + (p.σ^2 + (p.μ-q.μ)^2)/(2*q.σ^2) - 1//2
end

function RL.actions(e::AST_distributional)
    N = 2 * length(actions(e.env))
    return Box(fill(-Inf32,N), fill(Inf32,N))
end

RL.observations(e::AST_distributional) = product(Box([0], [Inf32]), observations(e.env))

RL.observe(e::AST_distributional) = [e.info[:steps][end]; observe(e.env)]

RL.terminated(e::AST_distributional) = terminated(e.env) || time_limit(e)

time_limit(e::AST_distributional) = e.info[:steps][end] >= e.n_steps

end