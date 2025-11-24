"""
DataAug Platform - A PySpark-based data augmentation framework for trajectory data.
"""

from .augmentations.base_augmentation import (
    Augmentation,
    local_aug,
    global_aug,
)
from .dataset import SparkIterableDataset
from .ingestion import load_hdf5_group, hdf5_to_rdd, read_hdf5_metadata, write_trajectories_to_hdf5
from .pipeline import Pipeline

__all__ = [
    # augmentation
    "Augmentation",
    "local_aug",
    "global_aug",
    # dataset
    "SparkIterableDataset",
    # ingestion
    "load_hdf5_group",
    "hdf5_to_rdd",
    "read_hdf5_metadata",
    "write_trajectories_to_hdf5",
    # Pipeline
    "Pipeline",
]

# Optional imports - only available if mimicgen dependencies are installed
try:
    from .augmentations.mimicgen import MimicGenAugmentation
    __all__.append("MimicGenAugmentation")
except ImportError:
    pass

__version__ = "0.1.0"
