"""
Test script to demonstrate running global augmentations multiple times using Spark.
"""

from pyspark.sql import SparkSession
from dataaug_platform import Augmentation, local_aug, global_aug, Pipeline
import numpy as np


class SimpleGlobalAugmentation(Augmentation):
    """Simple global augmentation that creates one new trajectory per run."""

    def __init__(self, times=1, keep_original=True):
        """
        Initialize the augmentation.

        Args:
            times: Number of times to run this augmentation (default: 1).
            keep_original: Whether to keep original trajectories (default: True).
        """
        super().__init__(times=times, keep_original=keep_original)

    @global_aug
    def apply(self, trajs):
        """
        Create a single averaged trajectory from all input trajectories.
        Each run will produce a different result due to randomness in averaging.
        """
        if not trajs:
            return []

        # Create an averaged trajectory with some randomness
        avg_traj = {}
        for key in trajs[0].keys():
            values = [traj[key] for traj in trajs if key in traj]
            if values and isinstance(values[0], np.ndarray):
                # Add small random variation to show different runs produce different results
                base_avg = np.mean(values, axis=0)
                noise = np.random.normal(0, 0.1, base_avg.shape)
                avg_traj[key] = base_avg + noise
            else:
                avg_traj[key] = values[0] if values else None

        return [avg_traj]


def test_multiple_runs():
    """Test running global augmentation multiple times."""

    spark = SparkSession.builder.appName("TestMultipleRuns").getOrCreate()

    # Create sample data
    sample_data = [
        {"x": np.array([1.0, 2.0, 3.0]), "y": np.array([4.0, 5.0, 6.0])},
        {"x": np.array([2.0, 3.0, 4.0]), "y": np.array([5.0, 6.0, 7.0])},
        {"x": np.array([3.0, 4.0, 5.0]), "y": np.array([6.0, 7.0, 8.0])},
    ]

    print("\n" + "=" * 60)
    print("Testing global augmentation with times=10")
    print("=" * 60)

    num_times = 5
    pipeline2 = Pipeline(spark)
    pipeline2.add(SimpleGlobalAugmentation(times=num_times, keep_original=False))
    result2 = pipeline2.run(sample_data)
    results2 = result2.collect()
    print(results2)

    print(f"Input: {len(sample_data)} trajectories")
    print(f"Output: {len(results2)} trajectories")
    print(f"Expected: {num_times}")

    spark.stop()


if __name__ == "__main__":
    test_multiple_runs()
