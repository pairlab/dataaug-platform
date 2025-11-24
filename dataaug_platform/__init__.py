"""
DataAug Platform - A PySpark-based data augmentation framework for trajectory data.
"""

from .augmentations.base_augmentation import (
    Augmentation,
    local_aug,
    global_aug,
)
from .pipeline import Pipeline
from .ingestion import load_hdf5_group, hdf5_to_rdd, read_hdf5_metadata, write_trajectories_to_hdf5

__all__ = [
    "Augmentation",
    "local_aug",
    "global_aug",
    "Pipeline",
    "load_hdf5_group",
    "hdf5_to_rdd",
    "read_hdf5_metadata",
    "write_trajectories_to_hdf5",
]

# Optional imports - only available if mimicgen dependencies are installed
try:
    from .augmentations.mimicgen import MimicGenAugmentation
    __all__.append("MimicGenAugmentation")
except ImportError:
    pass

__version__ = "0.1.0"
