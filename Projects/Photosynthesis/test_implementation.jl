using RCall
@rimport plantecophys

include("PlantEcoPhys_RL.jl")
using .PlantEcoPhys_RL

function get_R_info(x::PlantEcoPhys_RL.ExternalPhotosynParams)
    Robj = plantecophys.PhotosynEB(VPD = x.VPD, PPFD = x.PPFD, Tair = x.Tair, Wind = x.Wind)
    df = rcopy(Robj)
    A = df.ALEAF[]
    E = df.ELEAF[]
    GS = df.GS[]
    return (; A, E, GS)
end

df = vcat([rcopy(plantecophys.PhotosynEB(VPD = x.VPD, PPFD = x.PPFD, Tair = x.Tair, Wind = x.Wind)) for x in ext_traj]...)


ext_traj = PlantEcoPhys_RL.get_externaltraj()

internal = PlantEcoPhys_RL.InternalPhotosynParams()

A_err = 0.
A_ext = nothing
E_err = 0.
E_ext = nothing

for ext in ext_traj
    (A, E, GS) = get_R_info(ext)
    internal.GS = GS
    (A′, E′) = PlantEcoPhys_RL.PhotosynEB(internal, ext)
    if abs(A′-A) > A_err
        A_err = abs(A′-A)
        A_ext = ext
    end
    if abs(E′-E) > E_err
        E_err = abs(E′-E)
        E_ext = ext
    end
end

ext = A_ext
(A, E, GS) = get_R_info(ext)
internal.GS = GS
(A′, E′) = PlantEcoPhys_RL.PhotosynEB(internal, ext)
abs(A′-A)/A

ext = E_ext
(A, E, GS) = get_R_info(ext)
internal.GS = GS
(A′, E′) = PlantEcoPhys_RL.PhotosynEB(internal, ext)
abs(E′-E)/E