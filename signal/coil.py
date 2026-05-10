import numpy as np
from dataclasses import dataclass, field
from typing import List

@dataclass
class Coil:
    position: np.ndarray      # (x, y, z) mm
    normal: np.ndarray        # normalize, direction vector
    radius: float             # mm
    current: float            # Amper
    phase: float              # rad


@dataclass
class CoilCollection:
    frequency: float            # kHz
    coils: List[Coil] = field(default_factory=list)

    def add_coil(self, position: np.ndarray, normal: np.ndarray, radius: float, current: float, phase: float) -> None:
        """
        Adds a new coil to the collection after normalizing the normal vector.
        """
        normal = normal / np.linalg.norm(normal)
        coil = Coil(position=position, normal=normal, radius=radius, current=current, phase=phase)
        self.coils.append(coil)

    def add_ring(self, n: int, ring_radius: float, coil_radius: float, 
                 center: np.ndarray, axis: np.ndarray, current: float, phase_shift: float):
        """
        n: number of coils
        ring_radius: radius of the ring
        coil_radius: radius of each coil
        center: center of the ring
        axis: axis of the ring
        current: current in each coil
        phase_shift: phase shift between coils (rad)
        """
        for i in range(n):
            angle = 2 * np.pi * i / n
            pos_offset = ring_radius * np.array([np.cos(angle), np.sin(angle), 0])
            position = center + pos_offset
            phase = i * phase_shift
            self.add_coil(position, axis, coil_radius, current, phase)

    @property
    def omega(self):
        return 2 * np.pi * self.frequency
    
    