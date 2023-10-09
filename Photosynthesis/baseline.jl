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


function baseline(df; 
    leafA = 0.002, # PhotonsynEnv().internal.TotalLeafArea
    discount = 0.99,
    N_steps = 1_000,
    Einit = 2_000.0
    )

    idx = rand(1:size(df,1))
    r = 0.0
    E = Einit
    steps = 0
    for _ in 1:N_steps
        steps += 1
        idx = mod1(idx+1, size(df,1))
        r += discount^(steps-1) * df[idx, :ALEAF]
        E += 1000 * (rand() > 0.9)
        E -= PlantEcoPhys_RL.transpiration2loss(df[idx, :ELEAF], leafA)
        E <= 0 && break
    end

    r, steps
end

ext_traj = PlantEcoPhys_RL.get_externaltraj()
df = vcat([rcopy(plantecophys.PhotosynEB(VPD = x.VPD, PPFD = x.PPFD, Tair = x.Tair, Wind = x.Wind)) for x in ext_traj]...)

data = [baseline(df; Einit=Inf, N_steps=24, discount=1.0) for _ in 1:1000]
mean(x->x[1], data) # undiscounted 163 units photosyn / day
mean(x->x[2], data)
std([x[1] for x in data]) / sqrt(1000)

# mean 87.5 mg / hour
# 2.1 g / day
87.5 * 24
df.ELEAF |> mean |> x->PlantEcoPhys_RL.transpiration2loss(x, leafA)
ans * 24  * 30


2000 .+ 1000*cumsum(rand(24*30).>0.9)




env = PhotonsynEnv(; Esoil_init = Inf, GS_max=0.5)
reset!(env)

df.GS |> extrema


function rand_mean24(env)
    reset!(env)
    r = 0.0
    for step in 1:24
        # terminated(env) && break
        a = 2*rand() - 1
        r += act!(env, a)
    end
    r
end

mean((x)->rand_mean24(env), 1:10_000) # mean 164 units / 24 hr


# Fi

df.Tleaf2

df.failed |> count
df.failed
df.VPD[df.failed]


df[1,:]
ext_traj[1]
plantecophys.LeafEnergyBalance(VPD = ext_traj[1].VPD, PPFD = 0.0, Tair = ext_traj[1].Tair, Wind = ext_traj[1].Wind)


# something going on with parameters
# 5276 / 8784 = 60% failed energy balance


ext_traj[1]

plantecophys.PhotosynEB(Tair = 17.5)
# photosyneb doesnt work if its cold

plantecophys.Photosyn(Tleaf = 17.5)

internal = PlantEcoPhys_RL.InternalPhotosynParams(; GS=0.2)
external = ext_traj[1]
PlantEcoPhys_RL.PhotosynEB(internal, external)
internal.Tleaf

plantecophys.PhotosynEB(VPD = external.VPD, PPFD = external.PPFD, Tair = external.Tair, Wind = external.Wind)

max_diff = 0.0
for GS in 0:0.01:0.5
    internal.GS = GS
    for external in ext_traj
        PlantEcoPhys_RL.PhotosynEB(internal, external)
        diff = abs(internal.Tleaf - external.Tair)
        max_diff = max(max_diff, diff)
    end
end
max_diff # max leaf-air temp difference of 3.8 C

VPD = [x.VPD for x in ext_traj]
plot(VPD)

8784/24
y = zeros(24, 366)
plot([1,2,3,4,5,6,7,8,9,10], rand(10,2))