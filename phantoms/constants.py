"""Attenuation and relative stopping power constants for the phantom simulations.

Linear attenuation coefficients are mass attenuation coefficients at 100 keV from the NIST
tables (Hubbell and Seltzer, "Tables of X-Ray Mass Attenuation Coefficients and Mass
Energy-Absorption Coefficients", v1.4, NIST, 2004, doi:10.18434/T4D01F) times the density.
Relative stopping powers (RSP) are the ratio of the proton electronic stopping power of the
material to that of water (mean excitation energy 78 eV), computed with ``G4BetheBlochModel``
for the materials defined in the pct-sim simulation (https://github.com/mhentz/pct-sim) by
its ``rsp`` binary (``source/tools/rsp``), built against Geant4 10.7.4 and run as
``rsp 200 250``. Newer Geant4 versions give slightly different values, because they
normalise the mass fractions of each material to sum to one, which 10.7.4 did not enforce.
"""

# 100 keV, mass attenuation coefficient [cm2/g] times density [g/cm3]:
# dry air, water, B-100 bone-equivalent plastic and aluminium
MU_AIR = 0.1541 * 1.20479e-3
MU_WATER = 0.1707 * 1
MU_BONE = 0.1793 * 1.45
MU_AL = 0.1704 * 2.699


# 200 MeV
RSP_AIR = 0
RSP_WATER = 1  # 4.492/4.492
# NIST PSTAR total mass stopping powers [MeV cm2/g] of B-100 and water, times the B-100 density
RSP_BONE = 4.199 / 4.492 * 1.45
RSP_AL = 2.1272  # Aluminium (G4_Al) in the 200 MeV column of pct-sim's rsp output
RSP_LUNG = 0.295189  # LN300 in the 200 MeV column of pct-sim's rsp output

# 250 MeV, the 250 MeV column of pct-sim's rsp output
RSP_MAP = {
    "water": 1,
    "LN300": 0.295123,
    "LN450": 0.452646,
    "AP6": 0.947469,
    "BR12": 0.973937,
    "WaterSolidCT": 1.00695,
    "LV1": 1.08051,
    "SR2": 1.07373,
    "CB2_30": 1.27773,
    "CB2_50": 1.44863,
    "SB3": 1.65103,
    "B200": 1.10523,
    "IB3": 1.08807,
}

MATERIALS_MAP = {
    0: "BR12",
    1: "SB3",
    2: "AP6",
    3: "LV1",
    4: "SR2",
    5: "WaterSolidCT",
    6: "LN300",
    7: "CB2_50",
    8: "LN450",
    9: "CB2_30",
    10: "WaterSolidCT",
    11: "IB3",
    12: "LV1",
    13: "AP6",
    14: "B200",
    15: "LN300",
}


# 100 keV x-rays, elemental mass attenuation coefficients from the NIST tables
mu_H = 2.944e-1  # cm2/g
mu_C = 1.514e-1
mu_N = 1.529e-1
mu_O = 1.551e-1
mu_Mg = 1.686e-1
mu_Si = 1.835e-1
mu_P = 1.865e-1
mu_Cl = 2.050e-1
mu_Ca = 2.571e-1

rho_LN300 = 0.3  # g/cm3
rho_LN450 = 0.49
rho_AP6 = 0.948
rho_BR12 = 0.98
rho_water_solid = 1.017
rho_LV1 = 1.092
rho_SR2 = 1.051
rho_CB2_30 = 1.334
rho_CB2_50 = 1.561
rho_SB3 = 1.823
rho_B200 = 1.149
rho_IB3 = 1.136

# mu in units of [1/cm]
mu_LN300 = (
    (8.46 * mu_H + 59.37 * mu_C + 1.96 * mu_N + 18.14 * mu_O + 11.19 * mu_Mg + 0.78 * mu_Si + 0.1 * mu_Cl)
    / 100
    * rho_LN300
)
mu_LN450 = (
    (8.47 * mu_H + 59.56 * mu_C + 1.97 * mu_N + 18.11 * mu_O + 11.21 * mu_Mg + 0.58 * mu_Si + 0.1 * mu_Cl)
    / 100
    * rho_LN450
)
mu_AP6 = (9.06 * mu_H + 72.29 * mu_C + 2.25 * mu_N + 16.27 * mu_O + 0.13 * mu_Cl) / 100 * rho_AP6
mu_BR12 = (
    (8.59 * mu_H + 70.1 * mu_C + 2.33 * mu_N + 17.9 * mu_O + 0.13 * mu_Cl + 0.95 * mu_Ca) / 100 * rho_BR12
)
mu_water_solid = (
    (8.0 * mu_H + 67.29 * mu_C + 2.39 * mu_N + 19.87 * mu_O + 0.14 * mu_Cl + 2.31 * mu_Ca)
    / 100
    * rho_water_solid
)
mu_LV1 = (
    (8.06 * mu_H + 67.01 * mu_C + 2.47 * mu_N + 20.01 * mu_O + 0.14 * mu_Cl + 2.31 * mu_Ca) / 100 * rho_LV1
)
mu_SR2 = (10.83 * mu_H + 72.54 * mu_C + 1.69 * mu_N + 14.86 * mu_O + 0.08 * mu_Cl) / 100 * rho_SR2
mu_CB2_30 = (
    (6.68 * mu_H + 53.47 * mu_C + 2.12 * mu_N + 25.61 * mu_O + 0.11 * mu_Cl + 12.01 * mu_Ca)
    / 100
    * rho_CB2_30
)
mu_CB2_50 = (
    (4.77 * mu_H + 41.61 * mu_C + 1.52 * mu_N + 32.0 * mu_O + 0.08 * mu_Cl + 20.02 * mu_Ca) / 100 * rho_CB2_50
)
mu_SB3 = (
    (3.41 * mu_H + 31.41 * mu_C + 1.84 * mu_N + 36.49 * mu_O + 0.04 * mu_Cl + 26.81 * mu_Ca) / 100 * rho_SB3
)
mu_B200 = (
    (6.65 * mu_H + 55.51 * mu_C + 1.98 * mu_N + 23.64 * mu_O + 3.24 * mu_P + 0.11 * mu_Cl + 8.87 * mu_Ca)
    / 100
    * rho_B200
)
mu_IB3 = (
    (6.67 * mu_H + 55.65 * mu_C + 1.96 * mu_N + 23.52 * mu_O + 3.23 * mu_P + 0.11 * mu_Cl + 8.86 * mu_Ca)
    / 100
    * rho_IB3
)

MU_MAP = {
    "water": MU_WATER,
    "LN300": mu_LN300,
    "LN450": mu_LN450,
    "AP6": mu_AP6,
    "BR12": mu_BR12,
    "WaterSolidCT": mu_water_solid,
    "LV1": mu_LV1,
    "SR2": mu_SR2,
    "CB2_30": mu_CB2_30,
    "CB2_50": mu_CB2_50,
    "SB3": mu_SB3,
    "B200": mu_B200,
    "IB3": mu_IB3,
}

MU_LUNG = mu_LN300
