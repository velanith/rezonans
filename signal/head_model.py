from dataclasses import dataclass, field
from typing import Dict
import numpy as np

from signal.constants import Tissue, TissueProps


@dataclass
class HeadModel:
    T_blood: float = 37.0 # °C
    T_body: float = 37.0 # °C
    T_ambient: float = 22.0 # °C
    tumor_center: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0])) # mm
    tumor_radius: float = field(default_factory=lambda: 40.0) # mm

    layers: Dict[str, Tissue] = field(default_factory=lambda: {
        "skin": Tissue.SKIN,
        "bone_cortical": Tissue.BONE_CORTICAL,
        "bone_cancellous": Tissue.BONE_CANCELLOUS,
        "csf": Tissue.CSF,
        "brain_gm": Tissue.BRAIN_GM,
        "brain_wm": Tissue.BRAIN_WM,
        "dura": Tissue.DURA,
        "blood": Tissue.BLOOD,
        "tumor_et": Tissue.TUMOR_ET,
        "tumor_necrotic": Tissue.TUMOR_NECROTIC,
        "edema": Tissue.EDEMA,
    })

    def get_props(self, layer_name: str) -> TissueProps:
        return self.layers.get(layer_name.lower()).value

    def is_in_tumor(self, position: np.ndarray) -> bool:
        return np.linalg.norm(position - self.tumor_center) <= self.tumor_radius

    

    

    
    