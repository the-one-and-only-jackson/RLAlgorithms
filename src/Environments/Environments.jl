module Environments

#=
Class control environments from Farma's gymnasium (successor to OpenAI's gym)
=#

using CommonRLInterface
using Parameters: @with_kw, @unpack
using StaticArrays: SA, SVector, MVector, @MVector
using Distributions

using ..Spaces
using ..CommonRLExtensions

include("mountaincar.jl")
export MountainCar

include("pendulum.jl")
export Pendulum

include("acrobot.jl")
export Acrobot

include("cartpole.jl")
export CartPole

function rk4(dydt, y0, t, args...)
    # if dydt=f(y, t) then t must be passed in args (i.e. appears in argument twice).
    # only returns y(t), used for discretization
    y_out = y0
    for i in length(t)-1
        dt = t[i+1] - t[i]
        k1 = dydt(y_out, args...)
        k2 = dydt(y_out + k1*dt/2, args...)
        k3 = dydt(y_out + k2*dt/2, args...)
        k4 = dydt(y_out + k3*dt, args...)
        y_out += (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    end
    return y_out
end


end