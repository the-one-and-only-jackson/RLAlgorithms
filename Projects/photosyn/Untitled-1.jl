include("PlantEcoPhys_RL.jl")
using .PlantEcoPhys_RL
using CommonRLInterface
using RLAlgorithms.MultiEnv
using RLAlgorithms.Algorithms
using RLAlgorithms.CommonRLExtensions
using Plots
using Statistics
using RCall
@rimport plantecophys

ext_traj = PlantEcoPhys_RL.get_externaltraj()

t = 24*200 .+ (1:24)
PPFD = 1*[x.PPFD for x in ext_traj][t]
Tair = [x.Tair for x in ext_traj][t]
VPD = PlantEcoPhys_RL.RHtoVPD.(50, Tair)
 
p_ppfd = plot(0:23, PPFD, label=false, xlabel="Time of Day [hour]", title="Flux Density [umol / m2 / s]")
p_Tair = plot(0:23, Tair, label=false, xlabel="Time of Day [hour]", title="Air Temperature [C]")
p_VPD = plot(0:23, VPD, label=false, xlabel="Time of Day [hour]", title="Vapor Pressure Deficit [kPa]")

p1, p2, p3 = [plot() for _ in 1:3]
for gsmodel in ["BBOpti","BBLeuning","BallBerry"]
    data = vcat([rcopy(plantecophys.PhotosynEB(gsmodel=gsmodel, PPFD=PPFD[ti], Tair=Tair[ti], VPD=VPD[ti])) for ti in 1:24]...)
    @assert !any(data.failed) "Failed leaf energy balance"
    p1 = plot!(p1, 0:23, data.GS; label=gsmodel, xlabel="Time of Day [hour]", title="Conductance [mol / m2 / s]")
    p2 = plot!(p2, 0:23, data.ALEAF; label=gsmodel, xlabel="Time of Day [hour]", title="Rate of Photosynthesis [umol / m2 / s]")
    p3 = plot!(p3, 0:23, data.ELEAF; label=gsmodel, xlabel="Time of Day [hour]", title="Transpiration [mmol / m2 / 2]")
end
plot(p_ppfd, p1, p_Tair, p2, p_VPD, p3, 
layout=(3,2), size=(1200,900), legend=:outerright)

savefig("sample_day.png")



data_ballberry = vcat([rcopy(plantecophys.PhotosynEB(gsmodel="BallBerry", PPFD=PPFD[ti], Tair=Tair[ti], VPD=VPD[ti])) for ti in 1:24]...)
sum(data_ballberry.ALEAF)
sum(data_ballberry.ELEAF)

gs = plot(0:23, data_ballberry.GS, label=false, xlabel="Time of Day [hour]", title="Conductance [mol / m2 / s]")
dTleaf = plot(0:23, data_ballberry.Tleaf2 - data_ballberry.Tleaf, label=false, xlabel="Time of Day [hour]", title="Tleaf2 - Tleaf [C]")
dTair = plot(0:23, data_ballberry.Tleaf2 - data_ballberry.Tair, label=false, xlabel="Time of Day [hour]", title="Tleaf2 - Tair [C]")
plot(p_Tair, gs, dTleaf, dTair, layout=(2,2), size=(800,800))
savefig("sample_day_dTemp.png")

data_ballberry = vcat([rcopy(plantecophys.Photosyn(GS=0.03, PPFD=PPFD[ti], Tleaf=Tair[ti], VPD=VPD[ti])) for ti in 1:24]...)
sum(data_ballberry.ALEAF)
sum(data_ballberry.ELEAF)

ti = 10

A_sum, E_sum = 0.0, 0.0
for ti in 1:24
    internal = PlantEcoPhys_RL.InternalPhotosynParams(; GS=data_ballberry.GS[ti])
    external = PlantEcoPhys_RL.ExternalPhotosynParams(; PPFD=PPFD[ti], Tair=Tair[ti], VPD=VPD[ti])
    A,E = PlantEcoPhys_RL.PhotosynEB(internal, external)
    A_sum += A
    E_sum += E
end
A_sum, E_sum
39.9 * (60*60*24/1000/1000)
3.5 * (60*60*24/1000/1000*18.02)

(0:0.001:0.2)[argmin(abs.(A_vec .- 39.9))]

(0:0.001:0.2)[argmin(abs.(E_vec .- 3.5))]




A_vec = Float64[]
E_vec = Float64[]
for GS in 0:0.001:0.2
    A_sum, E_sum = 0.0, 0.0
    for ti in 1:24
        internal = PlantEcoPhys_RL.InternalPhotosynParams(; GS)
        external = PlantEcoPhys_RL.ExternalPhotosynParams(; PPFD=PPFD[ti], Tair=Tair[ti], VPD=VPD[ti])
        A,E = PlantEcoPhys_RL.PhotosynEB(internal, external)
        A_sum += A
        E_sum += E
    end
    push!(A_vec, A_sum)
    push!(E_vec, E_sum)
end
plot(
    plot(0:0.005:0.2, 60*60*24*A_vec/1000/1000; label=false, xlabel="Conductance [mol / m2 / s]", ylabel="Carbon Gain [mol / m2]"),
    plot(0:0.005:0.2, 60*60*24*E_vec/1000/1000*18.02; label=false, xlabel="Conductance [mol / m2 / s]", ylabel="Water Loss [kg / m2]"),
    layout=(2,1),
    size=(600,600)
)
savefig("conductance_plot.png")

data_ballberry.GS[ti]
# 

function testfun(gs)
    A_sum, E_sum = 0.0, 0.0
    for ti in 1:24
        internal = PlantEcoPhys_RL.InternalPhotosynParams(; GS=rand()*gs)
        external = PlantEcoPhys_RL.ExternalPhotosynParams(; PPFD=PPFD[ti], Tair=Tair[ti], VPD=VPD[ti])
        A,E = PlantEcoPhys_RL.PhotosynEB(internal, external)
        A_sum += A
        E_sum += E
    end
    # A_sum
    E_sum
end

mean([testfun(0.022) for _ in 1:100])
