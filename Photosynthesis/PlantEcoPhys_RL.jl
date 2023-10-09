module PlantEcoPhys_RL

export PhotonsynEnv

using Parameters
using CommonRLInterface
using StaticArrays
using CSV, DataFrames
using Statistics: mean, std

using RLAlgorithms.Spaces: Box
using RLAlgorithms.CommonRLExtensions

@consts begin
    Boltz       = 5.67e-8   # w M-2 K-4
    Emissivity  = 0.95      # -
    LatEvap     = 2.54      # MJ kg-1
    CPAIR       = 1010.0    # J kg-1 K-1
    H2OLV0      = 2.501e6   # J kg-1
    H2OMW       = 18e-3     # J kg-1
    AIRMA       = 29.e-3    # mol mass air (kg/mol)
    AIRDENS     = 1.204     # kg m-3
    UMOLPERJ    = 4.57
    DHEAT       = 21.5e-6   # molecular diffusivity for heat  
    RGas        = 8.314
    GCtoGW      = 1.57
end

@with_kw mutable struct InternalPhotosynParams{T<:Real} @deftype T
    GS = 0. # stomatal conductance (to H2O)
    Esoil = 0. # Soil water amount ()

    TotalLeafArea = 0.002 # total plant leaf area m^2
    
    alpha   = 0.24  # Quantum yield of electron transport (mol mol-1)
    theta   = 0.85  # Shape of light response curve
    Jmax    = 100.  # Maximum rate of electron transport at 25 degrees C (mu mol m-2 s-1)
    Vcmax   = 50.   # Maximum carboxylation rate at 25 degrees C (mu mol m-2 s-1)
    
    # Day respiration rate params
    Tleaf   = 25.   # Leaf temperature (degrees C)
    Rd0     = 0.92  # Day respiration rate at reference temperature (TrefR). 
    Q10     = 1.92  # Temperature sensitivity of Rd.
    TrefR   = 25.   # Reference temperature for Rd (Celcius).
    Rdayfrac= 1.0   # Ratio of Rd in the light vs. in the dark.
    @assert Rd0 > 0 "Rd0 must be a positive value"
    
    # Vcmax temperature response parameters
    EaV     = 58_550.
    EdVC    = 200_000.
    delsC   = 629.26
    
    # Jmax temperature response parameters
    EaJ     = 29_680.
    EdVJ    = 200_000.
    delsJ   = 631.88
    
    # If true, corrects input Vcmax and Jmax for actual Tleaf.
    # If false, assumes the provided Vcmax and Jmax are at the Tleaf provided.
    Tcorrect::Bool = true

    Wleaf   = 0.02      # leaf width (m) 
    StomatalRatio = 1.  # 2 for amphistomatous
    LeafAbs = 0.86      # in shortwave range, much less than PAR
end

@with_kw struct ExternalPhotosynParams{T<:Real} @deftype T
    VPD  = 1.5      # Vapour pressure deficit (kPa) (can be calulated from relative humidity)
    Ca   = 400.     # Atmospheric CO2 concentration (ppm)
    PPFD = 1500.    # Photosynthetic photon flux density ('PAR') (mu mol m-2 s-1)
    Patm = 100.     # Atmospheric pressure (kPa). Setting only Patm does not correct for atmospheric pressure effects on photosynthesis rates.
    Tair = 25.
    Wind = 2.    
end

struct2vec(s::ExternalPhotosynParams) = SA[s.VPD, s.Ca, s.PPFD, s.Patm, s.Tair, s.Wind]

function PhotosynEB(internal::InternalPhotosynParams, external::ExternalPhotosynParams; dT=0.1, Trange=15)
    # not exactly great... fix later
    # bad rootfinding
    E = 0. # mmol m-2 s-1
    err = Inf
    Tleaf = 0.
    for T in external.Tair .+ (-Trange:dT:Trange)
        internal.Tleaf = T
        (_err, _E) = LeafEnergyBalance(internal, external)
        if abs(_err) < err
            Tleaf = T
            err = abs(_err)
            E = _E
        end
    end
    internal.Tleaf = Tleaf
    
    Am = photosyn(internal, external)

    return Am, E
end

function LeafEnergyBalance(internal::InternalPhotosynParams, external::ExternalPhotosynParams)
    @unpack_InternalPhotosynParams internal
    @unpack_ExternalPhotosynParams external

    # Density of dry air
    AIRDENS = Patm*1000/(287.058 * Tk(Tair))

    # Latent heat of water vapour at air temperature (J mol-1)
    LHV = (H2OLV0 - 2.365e3 * Tair) * H2OMW

    # Const s in Penman-Monteith equation  (Pa K-1)
    SLOPE = (esat(Tair + 0.1) - esat(Tair)) / 0.1

    # Radiation conductance (mol m-2 s-1)
    Gradiation = 4*Boltz*Tk(Tair)^3 * Emissivity / (CPAIR * AIRMA)

    # See Leuning et al (1995) PC&E 18:1183-1200 Appendix E
    # Boundary layer conductance for heat - single sided, forced convection
    CMOLAR = Patm*1000 / (RGas * Tk(Tair))   # .Rgas() in package...
    Gbhforced = 0.003 * sqrt(Wind/Wleaf) * CMOLAR

    # Free convection
    GRASHOF = 1.6E8 * abs(Tleaf-Tair) * (Wleaf^3) # Grashof number
    Gbhfree = 0.5 * DHEAT * (GRASHOF^0.25) / Wleaf * CMOLAR

    # Total conductance to heat (both leaf sides)
    Gbh = 2*(Gbhfree + Gbhforced)

    # Heat and radiative conductance
    Gbhr = Gbh + 2*Gradiation

    # Boundary layer conductance for water (mol m-2 s-1)
    Gbw = StomatalRatio * 1.075 * Gbh  # Leuning 1995
    gw = GS*Gbw/(GS + Gbw)

    # Rnet
    Rsol = 2*PPFD/UMOLPERJ   # W m-2

    # Isothermal net radiation (Leuning et al. 1995, Appendix)
    ea = esat(Tair) - 1000*VPD
    ema = 0.642*(ea/Tk(Tair))^(1/7)
    Rnetiso = LeafAbs*Rsol - (1 - ema)*Boltz*Tk(Tair)^4 # isothermal net radiation

    # Isothermal version of the Penmon-Monteith equation
    GAMMA = CPAIR*AIRMA*Patm*1000/LHV
    ET = (1/LHV) * (SLOPE * Rnetiso + 1000*VPD * Gbh * CPAIR * AIRMA) / (SLOPE + GAMMA * Gbhr/gw)

    # Latent heat loss
    lambdaET = LHV * ET

    # Heat flux calculated using Gradiation (Leuning 1995, Eq. 11)
    Y = 1/(1 + Gradiation/Gbh)
    H2 = Y*(Rnetiso - lambdaET)

    # Leaf-air temperature difference recalculated from energy balance.
    # (same equation as above!)
    Tleaf2 = Tair + H2/(CPAIR * AIRDENS * (Gbh/CMOLAR))

    # Difference between input Tleaf and calculated, this will be minimized.
    EnergyBal = Tleaf - Tleaf2

    ELEAFeb = 1000*ET

    return EnergyBal, ELEAFeb
end

function photosyn(internal::InternalPhotosynParams, external::ExternalPhotosynParams; Ap=1_000) # i dont understand Ap
    @unpack_InternalPhotosynParams internal
    @unpack_ExternalPhotosynParams external

    (GS <= 0) && return zero(GS)

    GammaStar = TGammaStar(Tleaf, Patm)
    Km = TKm(Tleaf, Patm)

    if Tcorrect
        Vcmax *= TVcmax(Tleaf, EaV, delsC, EdVC)
        Jmax *= TJmax(Tleaf, EaJ, delsJ, EdVJ)
    end

    # Electron transport rate
    VJ = Jfun(PPFD, alpha, Jmax, theta)/4

    # Day respiration rate (mu mol m-2 s-1)
    Rd = get_Rd(Tleaf, Rd0, Q10, TrefR, Rdayfrac)
    
    GC = GS / GCtoGW

    A = 1/GC
    Bc = (Rd - Vcmax)/GC - Ca - Km
    Cc = Vcmax * (Ca - GammaStar) - Rd * (Ca + Km)
    
    Bj = (Rd - VJ)/GC - Ca - 2*GammaStar
    Cj = VJ * (Ca - GammaStar) - Rd * (Ca + 2*GammaStar)

    Ac = QUADM(A,Bc,Cc) # Rubisco activity is limiting
    Aj = QUADM(A,Bj,Cj) # Electron transport is limiting

    (Ac, Aj) = (Ac, Aj) .+ Rd # Net to gross

    Am = -QUADP(1 - 1e-04, Ac+Aj, Ac*Aj)
    if Am > Ap
        Am = -QUADP(1 - 1e-07, Am+Ap, Am*Ap)
    end

    Am -= Rd # gross to net ? 

    return Am
end


## === Various Utilities === ##
Jfun(PPFD, alpha, Jmax, theta) = (alpha*PPFD + Jmax - sqrt((alpha*PPFD + Jmax)^2 - 4*alpha*theta*PPFD*Jmax))/(2*theta)

# Arrhenius
arrh(Tleaf, Ea) = exp((Ea * (Tk(Tleaf) - Tk(25))) / (Tk(25) * RGas * Tk(Tleaf))) 

TGammaStar(Tleaf, Patm; Egamma=37830.0, value25=42.75) = value25*arrh(Tleaf,Egamma)*Patm/100

get_Rd(Tleaf, Rd0, Q10, TrefR, Rdayfrac) =  Rdayfrac*Rd0*Q10^((Tleaf-TrefR)/10)

function TKm(Tleaf, Patm;
    Oi = 210,       # O2 concentration (mmol mol-1)
    Ec = 79430.0,   # activation energy for Kc 
    Eo = 36380.0,   # activation energy for Ko
    Kc25 = 404.9,   # Kc at 25C
    Ko25 = 278.4    # Ko at 25C
    )
  
    Oi = Oi * Patm / 100

    Ko = Ko25 * arrh(Tleaf, Eo)
    Kc = Kc25 * arrh(Tleaf, Ec)
    Km = Kc * (1 + Oi / Ko)
  
    return(Km)
end

# Vcmax temperature response (Arrhenius)
function TVcmax(Tleaf, EaV, delsC, EdVC)
    ret = exp((Tleaf-25)*EaV/(RGas*Tk(Tleaf)*Tk(25)))
    if EdVC > 0
        f(x) = 1+exp((delsC*Tk(x)-EdVC)/(RGas*Tk(x)))
        ret *= f(25)/f(Tleaf)
    end
    ret
end

# Jmax temperature response (Arrhenius)
function TJmax(Tleaf, EaJ, delsJ, EdVJ)
    J1 = 1+exp((Tk(25)*delsJ-EdVJ)/RGas/Tk(25))
    J2 = 1+exp((Tk(Tleaf)*delsJ-EdVJ)/RGas/Tk(Tleaf))
    exp(EaJ/RGas*(1/Tk(25) - 1/Tk(Tleaf)))*J1/J2
end

function esat(TdegC, Pa=101)
    temp1 = 1.0007 + 3.46 * 10^-8 * Pa * 1000
    temp2 = exp(17.502 * TdegC/(240.97 + TdegC))
    return 611.21 * temp1 * temp2
end

QUADM(a,b,c) = quadratic(a,b,c)[1]
QUADP(a,b,c) = quadratic(a,b,c)[2]
function quadratic(a,b,c)
    @assert !(iszero(a) && iszero(b)) "No/inf soln"
    iszero(a) && return (-c/b, -c/b)
    disc = b^2 - 4*a*c
    @assert disc>=0 "Imaginary roots"
    return (-b-sqrt(disc), -b+sqrt(disc)) ./ (2*a)
end

Tk(x) = x + 273.15 # celcius to kelvin

function RHtoVPD(RH, TdegC, Pa=101)
	esatval = esat(TdegC, Pa)
	e = (RH/100) * esatval
	VPD = (esatval - e)/1000
    return VPD
end

### RL ###

@with_kw mutable struct PhotonsynEnv{A,B} <: AbstractEnv
    internal::A = InternalPhotosynParams()
    external_trajectory::B = get_externaltraj()
    ext_mean::SVector{6,Float64} = SVector{6,Float64}( mean(stack(struct2vec.(external_trajectory)); dims=2) )
    ext_std::SVector{6,Float64} = SVector{6,Float64}( std(stack(struct2vec.(external_trajectory)); dims=2) )
    idx::Int = rand(1:length(external_trajectory))
    step::Int = 1
    Esoil_init_max::Float64 = 2000.0
    GS_min::Float64 = 0.0
    GS_max::Float64 = 0.5
    n_obs::Int = 1 # weather model is partially observable
    max_steps::Int = 24*10
end

function CommonRLInterface.reset!(env::PhotonsynEnv)
    env.step = 1
    env.idx = rand(1:length(env.external_trajectory))
    env.internal = InternalPhotosynParams()
    env.internal.Esoil = env.Esoil_init_max * rand()
    nothing
end

CommonRLInterface.act!(env::PhotonsynEnv, a::AbstractArray) = act!(env, a[])
function CommonRLInterface.act!(env::PhotonsynEnv, a::Real)
    # action should be bounded? [0,1] (not 100% on this upper lim, just gut feel)
    # actor critic bounded [-1,1]

    internal, external = env.internal, env.external_trajectory[env.idx]

    a_01 = (1 + clamp(a, -one(a), one(a)))/2
    internal.GS = env.GS_min + (env.GS_max - env.GS_min) * a_01
    
    A,E = PhotosynEB(internal, external)
    # internal.Esoil += 1_000.0 * (rand() > 0.9)
    internal.Esoil -= E # transpiration2loss(E, internal.TotalLeafArea)
    env.step += 1
    env.idx = mod1(env.idx, length(env.external_trajectory))
    return A
end

function transpiration2loss(E, area; dt=3600, molarmass=18) # 18 g / mol
    mass = E * dt * molarmass * area  # mg
end

CommonRLInterface.terminated(env::PhotonsynEnv) = env.internal.Esoil <= 0 || env.step >= env.max_steps
CommonRLExtensions.truncated(env::PhotonsynEnv) = false

function CommonRLInterface.observe(env::PhotonsynEnv)
    o_E = env.internal.Esoil/env.Esoil_init_max
    o_s = (env.max_steps-env.step)/env.max_steps
    observation = Float32[o_E, o_s]
    for i in 0:env.n_obs-1
        idx = mod1(env.idx - i, length(env.external_trajectory))
        y = struct2vec(env.external_trajectory[idx])
        append!(observation, (y .- env.ext_mean) ./ (1f-7 .+ env.ext_std))
    end
    return observation
end

function get_externaltraj()
    data = CSV.read("Photosynthesis/cs01_daily.csv", DataFrame)
    scale_fun(min,max,t) = min + (max-min) * cos(pi*(t-12)/24)^2
    ext_traj = ExternalPhotosynParams{Float64}[]
    for row in eachrow(data[(data.YEAR .== 2020), :])
        for hour in 0:23
            Tair = scale_fun(row.TMIN, row.TMAX, hour)
            PPFD = scale_fun(0, row.PPFD, hour)
            Wind = row.WIND
            VPD = RHtoVPD(scale_fun(row.RHMIN,row.RHMAX,hour), Tair)        
            params = ExternalPhotosynParams(; Tair, PPFD, Wind, VPD)
            push!(ext_traj, params)
        end
    end
    return ext_traj
end

CommonRLExtensions.observations(env::PhotonsynEnv) = Box(fill(-Inf32, 6*env.n_obs+2), fill(Inf32, 6*env.n_obs+2))
CommonRLExtensions.actions(::PhotonsynEnv) = Box(SA[-1f0], SA[1f0])

end