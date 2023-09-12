abstract type AbstractMultiWrapper <: AbstractMultiEnv end

Wrappers.unwrapped(env::AbstractMultiWrapper) = unwrapped(wrapped_env(env))
Wrappers.unwrapped(env::AbstractMultiEnv) = env

macro forward_to_wrapped(f)
    return :($f(w::AbstractMultiWrapper, args...; kwargs...) = $f(wrapped_env(w), args...; kwargs...))
end

@forward_to_wrapped CommonRLInterface.reset!
# @forward_to_wrapped CommonRLInterface.actions
@forward_to_wrapped CommonRLInterface.observe
@forward_to_wrapped CommonRLInterface.act!
@forward_to_wrapped CommonRLInterface.terminated

@forward_to_wrapped CommonRLInterface.render
@forward_to_wrapped CommonRLInterface.state
@forward_to_wrapped CommonRLInterface.setstate!
@forward_to_wrapped CommonRLInterface.valid_actions
@forward_to_wrapped CommonRLInterface.valid_action_mask
# @forward_to_wrapped CommonRLInterface.observations
# not straightforward to provide clone

@forward_to_wrapped CommonRLInterface.players
@forward_to_wrapped CommonRLInterface.player
@forward_to_wrapped CommonRLInterface.all_act!
@forward_to_wrapped CommonRLInterface.all_observe
@forward_to_wrapped CommonRLInterface.UtilityStyle

@forward_to_wrapped Base.length
@forward_to_wrapped MultiEnv.single_observations
@forward_to_wrapped MultiEnv.single_actions

@forward_to_wrapped CommonRLExtensions.truncated

CommonRLInterface.provided(f::Function, w::AbstractMultiWrapper, args...) = provided(f, wrapped_env(w), args...)
CommonRLInterface.provided(::typeof(CommonRLInterface.clone), w::AbstractMultiWrapper, args...) = false
