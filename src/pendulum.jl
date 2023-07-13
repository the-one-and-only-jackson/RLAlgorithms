using ControlSystems: ss, c2d, lqr
using Parameters: @with_kw
using LinearAlgebra: I, diagm
using StaticArrays: SA
using Distributions
using CommonRLInterface

using RLAlgorithms.Spaces

# http://www.it.uu.se/edu/course/homepage/regtek2/v20/Labs/labassignment.pdf
@with_kw struct Pend
    M = 2.4
    m = 0.23
    l = 0.36
    g = 9.81
    f_theta = 0.1*m*l
    u_lim = 10
end

function pend_nl(p::Pend, x, u, w)
    y, theta, d_y, d_theta = x

    u = clamp(u[], -p.u_lim, p.u_lim)

    mass_mat = SA[p.M+p.m p.m*p.l*cos(theta); p.m*p.l*cos(theta) p.m*p.l^2]
    forces = SA[0, -p.f_theta*d_theta] + SA[u, 0] + w
    RHS = p.m*p.l*sin(theta) * [d_theta^2, p.g]
    dd_x = mass_mat \ (RHS + forces)

    dxdt = [d_y; d_theta; dd_x]

    return dxdt
end

function pend_K(; p=Pend(), dt=0.01)
    A = [0 0 1 0; 
        0 0 0 1; 
        0 -p.m*p.g/p.M 0 p.f_theta*p.m/p.M;
        0 (p.M+p.m)*p.g/(p.l*p.M) 0 -(p.M+p.m)*p.f_theta/(p.M*p.m*p.l^2)]
    B = [0, 0, 1/p.M, -1/(p.l*p.M)]

    sys = ss(A,B,I,zeros(4,1))
    sys_d = c2d(sys, dt)

    Q = diagm([1,1,0,0])
    R = 1
    K = lqr(sys_d, Q, R)

    return K
end

@with_kw mutable struct PendSim <: AbstractEnv
	t::Float64 = 0.0
    dt::Float64 = 0.01
    p::Pend = Pend()
    x::Vector{Float64} = zeros(4)
    K::Matrix{Float64} = pend_K(; p, dt)
    noise::Distribution = Normal(0, 0.1f0)
    x0::Distribution = MvNormal([1, 0.1, 1, 1])
end

CommonRLInterface.act!(env::PendSim, a::Vector) = act!(env, a[])
function CommonRLInterface.act!(env::PendSim, a::Number)
    w = [0; a]
    u = -env.K * env.x
    env.x .+= env.dt * pend_nl(env.p, env.x, u, w)
    env.t += env.dt
    return 0
end

function CommonRLInterface.reset!(env::PendSim)
    env.t = 0.0
    env.x .= rand(env.x0)
    nothing
end

CommonRLInterface.actions(env::PendSim) = DistributionSpace(env.noise)

CommonRLInterface.observations(env::PendSim) = Box(-Inf*ones(5), Inf*ones(5))

CommonRLInterface.terminated(env::PendSim) = abs(env.x[2]) > 0.3

CommonRLInterface.observe(env::PendSim) = [env.t; env.x] .|> Float32