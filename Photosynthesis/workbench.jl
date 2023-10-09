include("onefile.jl")

internal = InternalPhotosynParams(; GS=0.01)
external = ExternalPhotosynParams()

PhotosynEB(internal, external); internal.Tleaf

[LeafEnergyBalance(internal, external)[1] for internal.Tleaf in external.Tair .+ (-15:15)]

#=
# install R
# install plantecophys from the R environment (not in julia)

add RCall

using RCall
@rimport plantecophys
plantecophys.Photosyn(GS=0.01)

# Ci    ALEAF   GS ELEAF      Ac       Aj   Ap   Rd VPD Tleaf  Ca
# 89.85713 1.975432 0.01  0.15 2.90013 3.073892 1000 0.92 1.5    25 400
#         Cc PPFD Patm
# 89.85713 1500  100

=#

using RCall
@rimport plantecophys
plantecophys.PhotosynEB()
# GS = 0.2
# Tleaf = 28.2
plantecophys.Photosyn(Tleaf=30)
plantecophys.Photosyn(Tleaf=30, GS=0.1963547)


include("onefile.jl")
external = ExternalPhotosynParams()
internal = InternalPhotosynParams(; GS=0.2014992, Tleaf=25.)
photosyn(internal, external)
photosyn_parts(internal, external)

internal.Tleaf

v = [LeafEnergyBalance(internal, external) for internal.Tleaf in external.Tair .+ (-15:15)]
plot(external.Tair .+ (-15:15), [x[1] for x in v])
using Plots


Trange = 15.
dT = 0.1
internal.Tleaf = argmin(external.Tair .+ (-Trange:dT:Trange)) do Tleaf
    internal.Tleaf = Tleaf
    println(internal.Tleaf)
    return LeafEnergyBalance(internal, external)[1]^2
end
internal.Tleaf

internal.Tleaf=25.

for internal.Tleaf in 0:10
    println()
end

internal.Tleaf


using CommonRLInterface
using StaticArrays

actions = Box(SA[0f0], SA[Inf32])

mutable struct PhotonsynEnv
    internal::InternalPhotosynParams
    external_trajectory::Vector{<:ExternalPhotosynParams}
    idx::Int
    step::Int
end

function CommonRLInterface.reset!(env::PhotonsynEnv; Esoil_init=0.0)
    env.step = 1
    env.idx = rand(1:length(external_trajectory))
    env.internal = InternalPhotosynParams()
    env.internal.Esoil = Esoil_init
    nothing
end

function CommonRLInterface.act!(env::PhotonsynEnv, a::Real)
    internal, external = env.internal, env.external_trajectory[env.idx]
    internal.GS = a
    A,E = PhotosynEB(internal, external)
    internal.Esoil -= transpiration2loss(E, internal.TotalLeafArea)
    env.step += 1
    env.idx = mod1(env.idx, length(env.external_trajectory))
    return A
end

function transpiration2loss(E, area; dt=3600, molarmass=18) # 18 g / mol
    mass = E * dt / molarmass * area  # mg/s
end

CommonRLInterface.terminated(env::PhotonsynEnv) = env.internal.Esoil <= 0
truncated(env::PhotonsynEnv) = env.step >= 1000

function CommonRLInterface.observe(env::PhotonsynEnv)
    ext = struct2vec(env.external_trajectory[env.idx])
    SA[ext; env.internal.Esoil]
end





# https://ameriflux.lbl.gov/data/aboutdata/data-variables/


using CSV, DataFrames

function get_externaltraj()
    data = CSV.read("cs01_daily.csv", DataFrame)
    scale_fun(min,max,t) = min + (max-min) * cos(pi*(t-12)/24)^2
    ext_traj = ExternalPhotosynParams{Float64}[]
    for row in eachrow(data[(data.YEAR .== 2020), :])
        for hour in 0:23
            Tair = scale_fun(row.TMIN, row.TMAX, hour)
            PPFD = 2 * scale_fun(0, row.PPFD, hour)
            Wind = row.WIND
            VPD = RHtoVPD(scale_fun(row.RHMIN,row.RHMAX,hour), Tair)        
            params = ExternalPhotosynParams(; Tair, PPFD, Wind, VPD)
            push!(ext_traj, params)
        end
    end
    return ext_traj
end

