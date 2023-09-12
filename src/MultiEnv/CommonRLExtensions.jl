using Tricks: static_hasmethod

function CommonRLExtensions.get_info(wrap::AbstractMultiWrapper)
    d = get_info(wrapped_env(wrap))

    if !static_hasmethod(info, Tuple{typeof(wrap)})
        return d
    end
    
    key = first(eachsplit(string(typeof(wrap)), "{"))
    if haskey(d, key)
        suffix = 2
        while haskey(d, key*string(suffix))
            suffix += 1
        end
        key *= string(suffix)
    end

    d[key] = info(wrap)

    return d
end

CommonRLInterface.provided(::typeof(info), e::VecEnv, args...) = provided(info, first(e.envs), args...)
function CommonRLExtensions.info(env::VecEnv)
    if provided(info, first(env.envs))
        Dict("env$i"=>info(env.envs[i]) for i in 1:length(env))
    else
        Dict{}()
    end
end


