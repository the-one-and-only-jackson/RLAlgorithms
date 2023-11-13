@with_kw struct CartPole{T<:AbstractFloat, D} <: AbstractEnv @deftype T
    g = 9.8
    mass_cart = 1.0
    mass_pole = 0.5
    total_mass = mass_cart + mass_pole
    length = 0.5
    pole_masslength = mass_pole * length
    force_mag = 10.
    dt = 0.02
    theta_max = 12*pi/180
    x_max = 2.4
    init_state::D = Product([Uniform(-0.05, 0.05) for _ in 1:4])
    state::MVector{4,T} = @MVector zeros(4)
    steps::MVector{1,Int} = @MVector [0] # force everthing else to be const (issue with Paramters.jl)
end

CommonRLInterface.actions(::CartPole) = Spaces.Discrete(2)
CommonRLInterface.observations(env::CartPole) = Box(
    -SA{Float32}[env.x_max, Inf32, env.theta_max, Inf32], 
    SA{Float32}[env.x_max, Inf32, env.theta_max, Inf32]
)

CommonRLInterface.terminated(env::CartPole) = abs(env.state[1])>env.x_max || abs(env.state[3])>env.theta_max
CommonRLExtensions.truncated(env::CartPole) = (env.steps[] >= 500)

CommonRLInterface.observe(env::CartPole) = copy(env.state)

function CommonRLInterface.reset!(env::CartPole, state=nothing)
    env.steps[] = 0
    env.state .= something(state, rand(env.init_state))
    nothing
end

CommonRLInterface.act!(env::CartPole, action::AbstractArray) = act!(env, action[])
function CommonRLInterface.act!(env::CartPole, action::Real)
    @assert action ∈ 1:2 "Action out of bounds"

    env.steps[] += 1

    x, x_dot, theta, theta_dot = env.state
    force = if action==1 env.force_mag else -env.force_mag end

    costheta, sintheta = cos(theta), sin(theta)

    temp = (force + env.pole_masslength * theta_dot^2 * sintheta) / env.total_mass
    thetaacc = (env.g * sintheta - costheta * temp) / (env.length * (4//3 - env.mass_pole * costheta^2 / env.total_mass))
    xacc = temp - env.pole_masslength * thetaacc * costheta / env.total_mass

    env.state .+= env.dt * SA[x_dot, xacc, theta_dot, thetaacc]

    return 1
end
