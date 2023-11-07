module Satellites

using Distributions
using CommonRLInterface
using LinearAlgebra
using StaticArrays
using Parameters
using Random

using RLAlgorithms.Spaces

mutable struct ExtendedKalman{N,M}
    x::MVector{N} # state
    P::MMatrix{N,N} # state covariance
    f::Function # dynamics
    h::Function # observation
    params::NamedTuple # parameters
    A::Union{Function, SMatrix{N,N}} # df/dx(x; params...)
    C::Union{Function, SMatrix{M,N}} # dh/dx(x; params...)
    Q::Union{Function, SMatrix{N,N}} # process noise covariance
    R::Union{Function, SMatrix{M,M}} # output noise covariance
end

function predict!(EKF::ExtendedKalman)	
	A = (EKF.A isa Function) ? EKF.A(EKF.x; EKF.params...) : EKF.A
    Q = (EKF.Q isa Function) ? EKF.Q(EKF.x; EKF.params...) : EKF.Q
    
    EKF.P .= A*EKF.P*A' .+ Q
    EKF.x .= EKF.f(EKF.x; EKF.params...)  
 
	nothing
end

function correct!(EKF::ExtendedKalman, y)	
    R = (EKF.R isa Function) ? EKF.R(EKF.x; EKF.params...) : EKF.R
    C = (EKF.C isa Function) ? EKF.C(EKF.x; EKF.params...) : EKF.C
    
    S = C*EKF.P*C' + R
    K = EKF.P*C'*inv(S)
    
	y_err = y - EKF.h(EKF.x; EKF.params...)
    EKF.x .+= K*y_err

    EKF.P .-= K*C*EKF.P

	nothing
end

mutable struct SUT{N,M}
    x::MVector{N} # state
    f::Function # dynamics
    h::Function # observation
    params::NamedTuple
    EKF::ExtendedKalman{N,M}
end

function mahalanobis(x1, x2, P)
    diff = x1 - x2
    return sqrt(diff ⋅ (P\diff))
end



function EKF_CWH(; proc_lim=1e-3, meas_lim=1)
    A = SA[0.999984255678768	3.68333526800308e-07	0.996030422777318	9.27905006635953e-05;
        -3.68333652882962e-07	0.999984255658331	-9.28287030043297e-05	0.996032051397510;
        -3.14886653984443e-05	7.35688373518416e-07	0.992060844121703	0.000185334350667068;
        -7.35688625350844e-07	-3.14887062039024e-05	-0.000185410653843331	0.992064097034754]

    B = SA[0.499999999637753	3.10804031269811e-05;
        -3.10804031269811e-05	0.499999998551013;
        0.999999998551013	9.32412093539221e-05;
        -9.32412093539221e-05	0.999999994204051]

    C = SA[0.0 1.0 0.0 0.0]

    function dynamics(x::MVector{4}, noise=zeros(2); kwargs...)
        return A*x + B*noise
    end
    
    function output(x::MVector{4}, noise=0.0; kwargs...)
        return x[2] + noise
    end

    G = proc_lim*B
    Q = G*G'
    R = @SMatrix [meas_lim^2]

    x0 = zeros(MVector{4})
    P0 = zeros(MMatrix{4,4})
    EKF = ExtendedKalman(x0, P0, dynamics, output, NamedTuple(), A, C, Q, R)

    x0 = zeros(MVector{4})
    return SUT(x0, dynamics, output, NamedTuple(), EKF)
end

function step!(sut::SUT, proc_noise, meas_noise)
    # all noise inputs bound to +- 1
    # scale here

    sut.x .= sut.f(sut.x, [proc_noise, 0.0]; sut.params...)
    y = sut.h(sut.x, meas_noise; sut.params...)
    
    predict!(sut.EKF)
    correct!(sut.EKF, y)

    return mahalanobis(sut.x, sut.EKF.x, sut.EKF.P)
end

function reset!(sut::SUT, x0=zeros(4,1), P=zeros(4,4))
    sut.x     .= x0
    sut.EKF.x .= x0
    sut.EKF.P .= sut.EKF.Q

    nothing
end

function observe(sut::SUT)
    # get lower triangular entries of covar
    LT = eltype(sut.EKF.P)[]
    N = size(sut.EKF.P)[2]
    for ii=1:N
        push!(LT, sut.EKF.P[ii:N,ii]...)
    end

    return [sut.x; sut.EKF.x; LT]

    # return vec([sut.x...; sut.EKF.x...]) # dont need covar for lin sys with fixed starting covar
end



@with_kw mutable struct CWHSim <: AbstractEnv
	sut::SUT = EKF_CWH()
    # env::Environment = Environment(:meas=>Normal(0,1), :proc=>Normal(0,1e-3)) # const Environment = Dict{Symbol, Sampleable}
    noise = MvNormal(zeros(Float32, 2), [1, 1f-3])
    dist::Float64 = 0.0
    fail_dist::Float64 = 5.0
end

function CommonRLInterface.act!(env::CWHSim, a::AbstractVector)
    env.dist = step!(env.sut, a[2], a[1])
    return 0
end

function CommonRLInterface.reset!(sim::CWHSim)
    sim.dist = 0.0
    x0 = rand.([Uniform(-1000, 1000); 
				Uniform(-1000, 1000);
				Uniform(-10, 10);
				Uniform(-10, 10)])
	reset!(sim.sut, x0)
    nothing
end

CommonRLInterface.actions(env::CWHSim) = DistributionSpace(env.noise)
CommonRLInterface.observations(::CWHSim) = Box(fill(-Inf,18), fill(Inf,18))
CommonRLInterface.terminated(env::CWHSim) = env.dist > env.fail_dist
CommonRLInterface.observe(env::CWHSim) = observe(env.sut)


end