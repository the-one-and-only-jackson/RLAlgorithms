module Pendulums

using ControlSystems: ss, c2d, lqr
using Parameters: @with_kw
using LinearAlgebra: I, diagm
using StaticArrays: SA
using Distributions: Distribution, Normal, MvNormal
using CommonRLInterface

using RLAlgorithms.Spaces

export CartPole_params, PendSim

# http://www.it.uu.se/edu/course/homepage/regtek2/v20/Labs/labassignment.pdf
@with_kw struct CartPole_params{T<:AbstractFloat} @deftype T
    M = 2.4
    m = 0.23
    l = 0.36
    g = 9.81
    f_theta = 0.1*m*l
    u_lim = 10.0
end

struct LQR_controller{T<:AbstractMatrix{<:Real}}
    K::T
end
(C::LQR_controller)(x) = -C.K*x

@with_kw struct PendSim{D1<:Distribution, D2<:Distribution, C<:LQR_controller, P<:CartPole_params, X<:AbstractVector{<:AbstractFloat}} <: AbstractEnv
    dt::Float64 = 0.01
    params::P = CartPole_params()
    controller::C = LQR_controller(pend_K(; params, dt))
    noise::D1 = Normal(0, 0.1)
    x0::D2 = MvNormal([1, 0.1, 1, 1])
    x::X = zeros(4)
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

CommonRLInterface.act!(env::PendSim, a::AbstractVector) = act!(env, a[])
function CommonRLInterface.act!(env::PendSim, a::Number)
    w = [0; a]
    u = env.controller(env.x)[]
    env.x .+= env.dt * pend_dynamics(env.params, env.x, u, w)
    return 0
end

function CommonRLInterface.reset!(env::PendSim)
    env.x .= rand(env.x0)
    nothing
end

CommonRLInterface.terminated(env::PendSim) = abs(env.x[2]) > 0.3
CommonRLInterface.actions(env::PendSim) = DistributionSpace(env.noise)
CommonRLInterface.observations(::PendSim) = Box(fill(-Inf32,4), fill(Inf32,4))
CommonRLInterface.observe(env::PendSim) = Float32.(env.x)

end
