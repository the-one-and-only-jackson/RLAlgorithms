@with_kw struct MountainCar{T<:AbstractFloat} <: AbstractEnv @deftype T
    min_action = -1.0f0
    max_action = 1.0f0
    min_position = -1.2f0
    max_position = 0.6f0
    max_speed = 0.07f0
    goal_position = 0.5f0
    goal_velocity = 0f0
    power = 0.0015f0
    state::MVector{2,T} = @MVector zeros(Float32, 2)
    steps::MVector{1,Int} = @MVector [0] # force everthing else to be const (issue with Paramters.jl)
end

function CommonRLInterface.observations(mc::MountainCar)
    lower = SA[mc.min_position, -mc.max_speed]
    upper = SA[mc.max_position, mc.max_speed]
    Box(lower, upper)
end

CommonRLInterface.observe(mc::MountainCar) = SA[mc.state[1]+0.3f0, mc.state[2]/0.07f0]

CommonRLInterface.actions(mc::MountainCar) = Box(SA[mc.min_action], SA[mc.max_action])

CommonRLInterface.act!(mc::MountainCar, action::AbstractArray) = act!(mc, action[])
function CommonRLInterface.act!(mc::MountainCar, action::Real)
    mc.steps[] += 1

    force = clamp(action, mc.min_action, mc.max_action)
    dv = force * mc.power - 0.0025 * cos(3 * mc.state[1])
    mc.state[2] = clamp(mc.state[2] + dv, -mc.max_speed, mc.max_speed)
    mc.state[1] = clamp(mc.state[1] + mc.state[2], mc.min_position, mc.max_position)

    if mc.state[1] == mc.min_position && mc.state[2] < 0
        mc.state[2] = 0
    end

    reward = -0.1*action^2 + 100*terminated(mc)

    return reward/10
end

function CommonRLInterface.terminated(mc::MountainCar)
    flag1 = mc.state[1] >= mc.goal_position
    flag2 = mc.state[2] >= mc.goal_velocity
    flag1 && flag2
end

CommonRLExtensions.truncated(mc::MountainCar) = mc.steps[] >= 1_000

function CommonRLInterface.reset!(mc::MountainCar, state=nothing)
    mc.steps[] = 0

    if isnothing(state)
        x0 = -0.5 + rand()/10 # uniform [-0.6, -0.4]
        v0 = 0.
        state = SA[x0, v0]
    end

    mc.state .= state

    nothing
end

