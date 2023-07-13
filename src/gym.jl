module Gym

using CommonRLInterface
import ..Spaces
using ..MultiEnv

export
    gym_env_list,
    GymEnv,
    VecGymEnv

const gym_env_list = """
Environment Sets:
    classic-control
    box2d

Discrete Environments:
    Acrobot-v1
    CartPole-v1
    MountainCar-v0
    LunarLander-v2

Continuous Environments:
    MountainCarContinuous-v0
    Pendulum-v1
    BipedalWalker-v3
    CarRacing-v2
"""

#=
using Pkg
Pkg.add("PyCall")
Pkg.add("Conda")
Pkg.build("PyCall")
import Conda
Conda.pip_interop(true, Conda.PYTHONDIR)
Conda.pip("install", "gymnasium")
Conda.pip("install", "gymnasium[classic-control]")
Conda.pip("install", "gymnasium[box2d]")
=#

# issue with syncrhonous vector environments
# async(error): gym.vector.make("CartPole-v1", 3, true)
# sync(fine): gym.vector.make("CartPole-v1", 3, false)


mutable struct GymEnv <: AbstractEnv
    const name::String
    const env
    const actions::Spaces.NumericArraySpace
    const observations::Spaces.NumericArraySpace
    obs::AbstractArray
    terminated::Bool
    truncated::Bool
end

function GymEnv(env_name="CartPole-v1")
    gym = pyimport("gymnasium")
    env = gym.make(env_name)

    action_space = env.action_space
    if pybuiltin(:isinstance)(action_space, gym.spaces.Discrete)
        init_act = convert(Int64, action_space.start)
        n_acts = convert(Int64, action_space.n)
        actions = Spaces.Discrete(init_act:(init_act+n_acts))
    elseif pybuiltin(:isinstance)(action_space, gym.spaces.Box)
        actions = Spaces.Box(action_space.low, action_space.high)
    else
        @assert false "GymEnv can only handle Discrete or Box action spaces"
    end

    observation_space = env.observation_space
    if pybuiltin(:isinstance)(observation_space, gym.spaces.Discrete)
        init_act = convert(Int64, observation_space.start)
        n_acts = convert(Int64, observation_space.n)
        observations = Spaces.Discrete(init_act:(init_act+n_acts))
    elseif pybuiltin(:isinstance)(observation_space, gym.spaces.Box)
        observations = Spaces.Box(observation_space.low, observation_space.high)
    else
        @assert false "GymEnv can only handle Discrete or Box action spaces"
    end

    obs, _ = env.reset()
    terminated = false
    truncated = false
    return GymEnv(env_name, env, actions, observations, obs, terminated, truncated)
end

CommonRLInterface.actions(env::GymEnv) = env.actions
CommonRLInterface.observations(env::GymEnv) = env.observations
CommonRLInterface.observe(env::GymEnv) = env.obs
CommonRLInterface.terminated(env::GymEnv) = env.terminated

function CommonRLInterface.reset!(env::GymEnv)
    env.obs, _ = env.env.reset()
    nothing
end

function CommonRLInterface.act!(env::GymEnv, a)
    env.obs, rewards, env.terminated, env.truncated, _ = env.env.step(a)
    return rewards
end



""" 
VecGymEnv
"""
struct VecGymEnv <: AbstractMultiEnv
    name::String
    env
    actions::Spaces.NumericArraySpace
    observations::Spaces.NumericArraySpace
    obs::AbstractArray
    terminated::Vector{Bool}
    truncated::Vector{Bool}
end

function VecGymEnv(env_name="CartPole-v1"; num_envs=1, async=false)
    gym = pyimport("gymnasium")
    env = gym.vector.make(env_name, num_envs, async) # true/async doenst work

    # Check if discrete or box
    action_space = env.single_action_space
    if pybuiltin(:isinstance)(action_space, gym.spaces.Discrete)
        init_act = convert(Int64, action_space.start)
        n_acts = convert(Int64, action_space.n)
        actions = Spaces.Discrete(init_act:(init_act+n_acts))
    elseif pybuiltin(:isinstance)(action_space, gym.spaces.Box)
        actions = Spaces.Box(action_space.low, action_space.high)
    else
        @assert false "GymEnv can only handle Discrete or Box action spaces"
    end

    if pybuiltin(:isinstance)(env.observation_space, gym.spaces.Discrete)
        init_act = convert(Int64, env.observation_space.start)
        n_acts = convert(Int64, env.observation_space.n)
        observations = Spaces.Discrete(init_act:(init_act+n_acts))
    elseif pybuiltin(:isinstance)(env.observation_space, gym.spaces.Box)
        act_low = env.observation_space.low
        act_high = env.observation_space.high
        observations = Spaces.Box(act_low, act_high)
    else
        @assert false "GymEnv can only handle Discrete or Box action spaces"
    end

    obs, _ = env.reset()

    # copying the data instead of view, is this good?
    # need to eachslice since python stores data in a stupid way
    vec_obs = [zeros(eltype(o), size(o)) for o in eachslice(obs; dims=1)]
    copyto!.(vec_obs, eachslice(obs; dims=1)) 

    done = falses(num_envs)
    
    return VecGymEnv(env_name, env, num_envs, actions, vec_obs, done)
end

CommonRLInterface.actions(env::VecGymEnv) = env.actions
CommonRLInterface.observe(env::VecGymEnv) = env.obs
CommonRLInterface.terminated(env::VecGymEnv) = env.done

function CommonRLInterface.reset!(env::VecGymEnv)
    obs, _ = env.env.reset()
    copyto!.(env.obs, eachslice(obs; dims=1))
    nothing
end

function CommonRLInterface.act!(env::VecGymEnv, a::AbstractArray{<:AbstractArray})
    @assert length(a)==env.num_envs "Error: Number of actions does not match number of envs"
    obs, rewards, done, truncated, _ = env.env.step(a)
    copyto!.(env.obs, eachslice(obs; dims=1))
    copyto!(env.done, done .|| truncated)
    return rewards
end
CommonRLInterface.act!(env::VecGymEnv, a::Matrix) = CommonRLInterface.act!(env, eachcol(a))

end