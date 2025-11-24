"""
DataAug Platform - A PySpark-based data augmentation framework for trajectory data.
"""

from .augmentations.base_augmentation import (
    Augmentation,
    local_aug,
    global_aug,
)
from .augmentations.mimicgen import MimicGenAugmentation
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
    "MimicGenAugmentation",
]

__version__ = "0.1.0"
