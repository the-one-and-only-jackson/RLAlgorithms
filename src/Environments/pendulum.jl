@with_kw struct Pendulum{T<:AbstractFloat, D<:Distribution} <: AbstractEnv @deftype T
    min_action = -2.
    max_action = 2.
    max_speed = 8.
    dt = 0.05
    g = 9.8
    m = 1.
    l = 1.
    max_steps::Int = 200
    init_state::D = Product([Uniform(-pi,pi), Uniform(-1, 1)])
    state::MVector{2,T} = @MVector zeros(2)
    steps::MVector{1,Int} = @MVector [0] # force everthing else to be const (issue with Paramters.jl)
end

CommonRLInterface.observations(env::Pendulum) = Box(SA[-1f0, -1f0, -Float32(env.max_speed)], SA[1f0, 1f0, Float32(env.max_speed)])

CommonRLInterface.observe(env::Pendulum) = SA{Float32}[cos(env.state[1]), sin(env.state[1]), env.state[2]/env.max_speed]

CommonRLInterface.actions(env::Pendulum) = Box(SA[Float32(env.min_action)], SA[Float32(env.max_action)])

CommonRLInterface.act!(env::Pendulum, action::AbstractArray) = act!(env, action[])
function CommonRLInterface.act!(env::Pendulum, action::Real)
    env.steps[] += 1

    force = clamp(2*action, env.min_action, env.max_action)

    theta, theta_dot = env.state

    theta_norm = mod2pi(pi+theta)-pi
    costs = theta_norm^2 + theta_dot^2/10 + force^2/1000

    theta_dot += env.dt * ( 3*env.g/(2*env.l)*sin(theta) + 3/(env.m*env.l^2)*force  ) 
    theta_dot = clamp(theta_dot, -env.max_speed, env.max_speed)
    theta += env.dt * theta_dot
    env.state .= SA[theta, theta_dot]

    return -costs
end

CommonRLInterface.terminated(::Pendulum) = false

CommonRLExtensions.truncated(env::Pendulum) = env.steps[] >= env.max_steps

function CommonRLInterface.reset!(env::Pendulum, state=nothing)
    env.steps[] = 0
    env.state .= something(state, rand(env.init_state))
    nothing
end

