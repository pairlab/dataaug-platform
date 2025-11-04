"""
DataAug Platform - A PySpark-based data augmentation framework for trajectory data.
"""

from .augmentation import (
    Augmentation,
    local_aug,
    global_aug,
)
from .pipeline import Pipeline
from .ingestion import load_hdf5_group, hdf5_to_rdd

__all__ = [
    "Augmentation",
    "local_aug",
    "global_aug",
    "Pipeline",
    "load_hdf5_group",
    "hdf5_to_rdd",
]

__version__ = "0.1.0"
