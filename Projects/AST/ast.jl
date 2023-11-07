module AST

using CommonRLInterface
using Parameters: @with_kw
using Distributions: loglikelihood, Normal, MvNormal, entropy
using Statistics: mean, std, cov
using LinearAlgebra: tr, logdet

using RLAlgorithms.Spaces: Box, product

export AST_distributional

const RL = CommonRLInterface

@with_kw mutable struct AST_distributional{E<:AbstractEnv} <: AbstractEnv
    env::E
    n_steps::Int
    terminal_cost::Float64 = 5*n_steps*loglikelihood(actions(env).d, mean(actions(env).d))
    step_vec::Vector{Int} = Int[]
    likelihood_vec::Vector{Float64} = Float64[]
    KL_vec::Vector{Float64} = Float64[]
    fail_vec::Vector{Bool} = Bool[]
    step::Int = 0
    likelihood::Float64 = 0.0
    KL::Float64 = 0.0
    fail::Bool = false
end

info(e::AST_distributional) = Dict(:steps=>e.step_vec, :likelihood=>e.likelihood_vec, :KL=>e.KL_vec, :fail=>e.fail_vec)

function RL.reset!(e::AST_distributional)
    if !iszero(e.step)
        steps_togo = e.n_steps - e.step
        expected_rew = -entropy(actions(e.env).d)
        e.likelihood += steps_togo * expected_rew

        foreach(push!, (e.step_vec, e.likelihood_vec, e.KL_vec, e.fail_vec), (e.step, e.likelihood, e.KL, e.fail))
    end

    e.step, e.likelihood, e.KL, e.fail = 0, 0.0, 0.0, false

    reset!(e.env)
    nothing
end

function RL.act!(e::AST_distributional, a)
    model = actions(e.env).d
    dist = unflatten(model, a)

    x = rand(dist)
    act!(e.env, x)

    kl = kl_divergence(dist, model)

    # logging
    e.step += 1
    e.likelihood += loglikelihood(model, x)
    e.KL += kl
    e.fail |= terminated(e.env)

    r = -kl - e.terminal_cost * time_limit(e)

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

function kl_divergence(p::MvNormal, q::MvNormal)
    du = mean(p) - mean(q)
    a1 = du' * (cov(q) \ du)
    a2 = tr(cov(q) \ cov(p))
    a3 = logdet(cov(q)) - logdet(cov(p))
    return (a1 - length(du) + a2 + a3)/2
end

function RL.actions(e::AST_distributional)
    N = 2 * length(actions(e.env))
    return Box(fill(-Inf32,N), fill(Inf32,N))
end

RL.observations(e::AST_distributional) = product(Box([0], [Inf32]), observations(e.env))

RL.observe(e::AST_distributional) = [e.step; observe(e.env)]

RL.terminated(e::AST_distributional) = terminated(e.env) || time_limit(e)

time_limit(e::AST_distributional) = e.step >= e.n_steps

end