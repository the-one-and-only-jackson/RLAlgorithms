module Pendulums

using ControlSystems: ss, c2d, lqr
using Parameters: @with_kw
using LinearAlgebra: I, diagm
using StaticArrays: SA
using Distributions: Distribution, Normal, MvNormal
using CommonRLInterface

using RLAlgorithms.Spaces

export CartPole_params, CartPole, PendSim

# http://www.it.uu.se/edu/course/homepage/regtek2/v20/Labs/labassignment.pdf
@with_kw struct CartPole_params
    M = 2.4
    m = 0.23
    l = 0.36
    g = 9.81
    f_theta = 0.1*m*l
    u_lim = 10
end

function pend_dynamics(p::CartPole_params, x::AbstractArray, u::Real, w::AbstractArray)
    @unpack_CartPole_params p

    y, theta, d_y, d_theta = x

    F = clamp(u, -u_lim, u_lim)

    mass_mat = SA[M+m m*l*cos(theta); m*l*cos(theta) m*l^2]
    forces = SA[0, -f_theta*d_theta] + SA[F, 0] + w
    RHS = m*l*sin(theta) * [d_theta^2, g]
    dd_x = mass_mat \ (RHS + forces)

    dxdt = SA[d_y, d_theta, dd_x...]

    return dxdt
end

@with_kw struct CartPole
    params::CartPole_params = CartPole_params()
    x::AbstractArray = zeros(4)
end

function step!(cp::CartPole, u::Real, w::AbstractArray{<:Real}, dt::Real)
    cp.x .+= dt * pend_dynamics(cp.params, cp.x, u, w)
    nothing
end

function set!(cp::CartPole, x)
    cp.x .= x
    nothing
end

function pend_K(; params=CartPole_params(), dt=0.01)
    @unpack_CartPole_params params

    A = [0 0 1 0; 
        0 0 0 1; 
        0 -m*g/M 0 f_theta*m/M;
        0 (M+m)*g/(l*M) 0 -(M+m)*f_theta/(M*m*l^2)]
    B = [0, 0, 1/M, -1/(l*M)]

    sys = ss(A,B,I,zeros(4,1))
    sys_d = c2d(sys, dt)

    Q = diagm([1,1,0,0])
    R = 1
    K = lqr(sys_d, Q, R)

    return K
end

struct LQR_controller
    K::AbstractMatrix{<:Real}
end
(C::LQR_controller)(x) = -C.K*x

@with_kw struct PendSim <: AbstractEnv
    dt::Float64 = 0.01
    p::CartPole = CartPole()
    controller = LQR_controller(pend_K(; p.params, dt))
    noise::Distribution = Normal(0, 0.1f0)
    x0::Distribution = MvNormal([1, 0.1, 1, 1])
end

CommonRLInterface.act!(env::PendSim, a::AbstractVector) = act!(env, a[])
function CommonRLInterface.act!(env::PendSim, a::Number)
    w = [0; a]
    u = env.controller(env.p.x)[]
    step!(env.p, u, w, env.dt)
    return 0
end

function CommonRLInterface.reset!(env::PendSim)
    set!(env.p, rand(env.x0))
    nothing
end

CommonRLInterface.actions(env::PendSim) = DistributionSpace(env.noise)
CommonRLInterface.observations(::PendSim) = Box(fill(-Inf32,4), fill(Inf32,4))
CommonRLInterface.terminated(env::PendSim) = abs(env.p.x[2]) > 0.3
CommonRLInterface.observe(env::PendSim) = Float32.(env.p.x)

end
