@with_kw struct Acrobot{T<:AbstractFloat, D} <: AbstractEnv @deftype T
    length_1 = 1.
    length_2 = 1.
    mass_1 = 1.
    mass_2 = 1.
    com_1 = 0.5
    com_2 = 0.5
    moi = 1.
    max_v1 = 4*pi
    max_v2 = 9*pi
    torque_noise = 0.
    dt = 0.2
    g = 9.8

    init_state::D = Product([Uniform(-0.1, 0.1) for _ in 1:4])
    avail_torque::SVector{3,T} = SA[-1., 0., 1.]

    state::MVector{4,T} = @MVector zeros(4)
    steps::MVector{1,Int} = @MVector [0] # force everthing else to be const (issue with Paramters.jl)
end

function CommonRLInterface.observations(env::Acrobot)
    upper = SA{Float32}[1, 1, 1, 1, env.max_v1, env.max_v2]
    Box(-upper, upper)
end

CommonRLInterface.observe(env::Acrobot) = SA{Float32}[cos(env.state[1]), sin(env.state[1]), cos(env.state[2]), sin(env.state[2]), env.state[3], env.state[4]]

CommonRLInterface.actions(::Acrobot) = Spaces.Discrete(3)

CommonRLInterface.act!(env::Acrobot, action::AbstractArray) = act!(env, action[])
function CommonRLInterface.act!(env::Acrobot, action::Real)
    env.steps[] += 1

    torque = env.avail_torque[action] + env.torque_noise * (2*rand()-1)

    next_state = rk4(env.state, [0, env.dt]) do y
        theta_1, theta_2, v_theta_1, v_theta_2 = y

        @unpack length_1, length_2, mass_1, mass_2, com_1, com_2, moi, max_v1, g = env

        d1 = 2*moi + mass_1 * com_1^2 + mass_2 * (length_1^2 + com_2^2 + 2 * length_1 * com_2 * cos(theta_2))

        d2 = moi + mass_2 * (com_2^2 + length_1 * com_2 * cos(theta_2))
        
        phi2 = mass_2 * com_2 * g * cos(theta_1 + theta_2 - pi/2)
        
        phi1 = phi2 - mass_2 * length_1 * com_2 * v_theta_2^2 * sin(theta_2)
        phi1 -= 2 * mass_2 * length_1 * com_2 * v_theta_2 * v_theta_1 * sin(theta_2)
        phi1 += (mass_1 * com_1 + mass_2 * length_1) * g * cos(theta_1 - pi/2)

        a_theta_2 = torque + d2 / d1 * phi1 - phi2
        a_theta_2 -= mass_2 * length_1 * com_2 * v_theta_1^2 * sin(theta_2) # can comment this line (NIPS)
        a_theta_2 /= mass_2 * com_2^2 + moi - d2^2 / d1

        a_theta_1 = -(d2 * a_theta_2 + phi1) / d1

        return SA[v_theta_1, v_theta_2, a_theta_1, a_theta_2]
    end

    env.state .= SA[
        clamp(next_state[1], -pi, pi),
        clamp(next_state[2], -pi, pi),
        clamp(next_state[3], -env.max_v1, env.max_v1),
        clamp(next_state[4], -env.max_v2, env.max_v2)
    ]

    return -1
end

function CommonRLInterface.terminated(env::Acrobot)
    (-cos(env.state[1]) - cos(env.state[2] + env.state[1])) > 1
end

CommonRLExtensions.truncated(env::Acrobot) = (env.steps[] >= 500)

function CommonRLInterface.reset!(env::Acrobot, state=nothing)
    env.steps[] = 0

    if isnothing(state)
        state = rand(env.init_state)
    end

    env.state .= state

    nothing
end
