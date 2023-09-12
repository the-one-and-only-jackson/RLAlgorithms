module CommonRLExtensions

using CommonRLInterface
using CommonRLInterface.Wrappers

using Tricks: static_hasmethod

export info, get_info, truncated

function info end

function get_info(env)
    if provided(info, env)
        Dict("env" => info(env))
    else
        Dict{}()
    end
end

function get_info(wrap::AbstractWrapper)
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

truncated(env) = false # should be auto default

end