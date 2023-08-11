module Utils

using Parameters: @with_kw
using Statistics: mean, std

export Logger, normalize

@with_kw struct Logger{T}
    log = Dict{Symbol, T}()
end

function (logger::Logger{Tuple{Vector, Vector}})(x_val; kwargs...)
    for (key, val) in kwargs
        x_vec, y_vec = get!(logger.log, key, (typeof(x_val)[], typeof(val)[]))
        push!(x_vec, x_val)
        push!(y_vec, val)
    end
    nothing
end

function (logger::Logger{Vector})(; kwargs...)
    for (key, val) in kwargs
        y_vec = get!(logger.log, key, typeof(val)[])
        push!(y_vec, val)
    end
    nothing
end

normalize(x::AbstractVector, eps = eltype(x)(1e-8)) = (x .- mean(x)) / (std(x) + eps)

end