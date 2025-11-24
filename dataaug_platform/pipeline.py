from abc import ABC, abstractmethod
from typing import Optional

from pyspark.sql import SparkSession

from .augmentations.base_augmentation import Augmentation

class Pipeline:
    """Manages a sequence of augmentations."""

    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = (
            spark or SparkSession.builder.appName("TrajectoryPipeline").getOrCreate()
        )
        self.augmentations = []
        self._base_rdd = None  # stored input RDD (for set_data-style use)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.spark is not None:
            self.spark.stop()

    def add(self, aug: "Augmentation"):
        """Add an augmentation to the pipeline."""
        self.augmentations.append(aug)
        return self  # enable chaining

    def set_data(self, data, cache=False):
        """
        Set the base data for the pipeline.

        `data` can be a Python list or an existing RDD.
        If `cache=True`, the RDD will be cached in memory.
        """
        sc = self.spark.sparkContext

        # list / iterable -> parallelize, RDD -> keep
        if hasattr(data, "context"):  # looks like an RDD
            rdd = data
        else:
            rdd = sc.parallelize(data)

        if cache:
            rdd = rdd.cache()

        self._base_rdd = rdd
        return self

    def run(self, data=None, use_stored_if_no_data=True):
        """
        Run all augmentations sequentially.

        - If `data` is provided:
            behaves like the old version: converts list -> RDD if needed, does NOT
            modify the stored base data.
        - If `data` is None and `use_stored_if_no_data` is True:
            uses the RDD set via `set_data(...)`.
        """
        sc = self.spark.sparkContext

        if data is not None:
            # old behavior: convert to RDD if needed
            if hasattr(data, "context"):  # RDD
                rdd = data
            else:  # list / iterable
                rdd = sc.parallelize(data)
        else:
            if not use_stored_if_no_data:
                raise ValueError("No data passed to run() and use_stored_if_no_data=False.")
            if self._base_rdd is None:
                raise ValueError(
                    "No data passed to run() and no data set via set_data(). "
                    "Call run(data=...) or set_data(...) first."
                )
            rdd = self._base_rdd

        # apply augmentations
        for aug in self.augmentations:
            rdd = aug._apply_rdd(rdd)

        return rdd
