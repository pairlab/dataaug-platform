from abc import ABC, abstractmethod
from pyspark.sql import SparkSession
from .augmentations.base_augmentation import Augmentation


class Pipeline:
    """Manages a sequence of augmentations."""

    def __init__(self, spark=None):
        self.spark = (
            spark or SparkSession.builder.appName("TrajectoryPipeline").getOrCreate()
        )
        self.augmentations = []

    def add(self, aug: Augmentation):
        """Add an augmentation to the pipeline."""
        self.augmentations.append(aug)
        return self  # enable chaining

    def run(self, data):
        """Run all augmentations sequentially."""
        sc = self.spark.sparkContext
        if not hasattr(data, "context"):  # convert list ? RDD if needed
            rdd = sc.parallelize(data)
        else:
            rdd = data

        for aug in self.augmentations:
            rdd = aug._apply_rdd(rdd)

        return rdd
