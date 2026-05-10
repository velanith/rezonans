# constants.py
# Source: IT'IS Foundation Tissue Properties Database
# https://itis.swiss/virtual-population/tissue-properties/database/
# Electromagnetic props based on C.Gabriel et al. 1996 via IFAC-CNR
# http://niremf.ifac.cnr.it/tissprop/

from dataclasses import dataclass
from enum import Enum

@dataclass(frozen=True)
class TissueProps:
    # Electromagnetic
    sigma: float              # S/m - electrical conductivity
    eps_r: float              # relative permittivity
    # Thermal
    density: float            # kg/m³
    heat_capacity: float      # J/kg/°C
    thermal_conductivity: float  # W/m/°C
    perfusion: float          # ml/min/kg
    heat_generation: float    # W/kg - metabolic heat generation


class Tissue(Enum):
    SKIN            = TissueProps(sigma=1.05e-3, eps_r=1.10e+3, density=1109, heat_capacity=3391, thermal_conductivity=0.37, perfusion=106.0,  heat_generation=1.65)
    BONE_CORTICAL   = TissueProps(sigma=2.11e-2, eps_r=2.04e+2, density=1908, heat_capacity=1313, thermal_conductivity=0.32, perfusion=10.0,   heat_generation=0.15)
    BONE_CANCELLOUS = TissueProps(sigma=8.46e-2, eps_r=3.87e+2, density=1178, heat_capacity=2274, thermal_conductivity=0.31, perfusion=30.0,   heat_generation=0.46)
    CSF             = TissueProps(sigma=2.00e+0, eps_r=1.09e+2, density=1007, heat_capacity=4096, thermal_conductivity=0.57, perfusion=0.0,    heat_generation=0.00)
    BRAIN_GM        = TissueProps(sigma=1.41e-1, eps_r=2.01e+3, density=1045, heat_capacity=3696, thermal_conductivity=0.55, perfusion=764.0,  heat_generation=15.54)
    BRAIN_WM        = TissueProps(sigma=8.68e-2, eps_r=1.29e+3, density=1041, heat_capacity=3583, thermal_conductivity=0.48, perfusion=212.0,  heat_generation=4.32)
    DURA            = TissueProps(sigma=5.02e-1, eps_r=2.90e+2, density=1174, heat_capacity=3364, thermal_conductivity=0.44, perfusion=380.0,  heat_generation=5.89)
    BLOOD           = TissueProps(sigma=7.10e-1, eps_r=4.93e+3, density=1050, heat_capacity=3617, thermal_conductivity=0.52, perfusion=10000.0, heat_generation=0.00)
    TUMOR_ET        = TissueProps(sigma=0.48,    eps_r=5000,    density=1046, heat_capacity=3630, thermal_conductivity=0.51, perfusion=559.0,  heat_generation=11.37)  # güncellenecek
    TUMOR_NECROTIC  = TissueProps(sigma=0.25,    eps_r=3500,    density=1046, heat_capacity=3630, thermal_conductivity=0.51, perfusion=559.0,  heat_generation=11.37)  # güncellenecek
    EDEMA           = TissueProps(sigma=0.20,    eps_r=3800,    density=1046, heat_capacity=3630, thermal_conductivity=0.51, perfusion=559.0,  heat_generation=11.37)  # güncellenecek