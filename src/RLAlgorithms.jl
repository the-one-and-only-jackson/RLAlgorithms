module RLAlgorithms

using Reexport

include("utils.jl")
@reexport using .Utils

include("Spaces/Spaces.jl")
@reexport using .Spaces

include("CommonRLExtensions/CommonRLExtensions.jl")
@reexport using .CommonRLExtensions

include("MultiEnv/MultiEnv.jl")
@reexport using .MultiEnv

include("Algorithms/Algorithms.jl")
@reexport using .Algorithms

include("Environments/Environments.jl")
@reexport using .Environments

end 
