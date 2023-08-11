module RLAlgorithms

include("utils.jl")

include("spaces.jl")

include("multienv.jl")

include("multienv_wrapper.jl")

# include("gym.jl")

include("ActorCritics.jl")
# using .ActorCritics

include("ppo.jl")
# using .PPO

end # module RLAlgorithms
