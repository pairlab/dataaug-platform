"""
Example demonstrating how to use user-defined augmentation classes.

This shows how to define augmentation classes with @local_aug and @global_aug decorators.
"""

from pyspark.sql import SparkSession
from dataaug_platform import Augmentation, local_aug, global_aug, Pipeline
import numpy as np


# ============================================================================
# Example 1: User-defined local augmentation class
# ============================================================================


class AddOffsetAugmentation(Augmentation):
    """Add a constant offset to all numeric values."""

    def __init__(self, offset=1.0):
        self.offset = offset

    @local_aug
    def apply(self, traj):
        """Process one trajectory at a time."""
        import numpy as np

        modified = traj.copy()
        for key, value in modified.items():
            if isinstance(value, np.ndarray):
                modified[key] = value + self.offset
        return modified


# ============================================================================
# Example 2: User-defined global augmentation class
# ============================================================================


class AverageTrajectoriesAugmentation(Augmentation):
    """Create a new trajectory by averaging all trajectories."""

    def __init__(self, times=1, keep_original=True):
        """
        Initialize the augmentation.

        Args:
            times: Number of times to run this augmentation (default: 1).
                   Each run processes the whole dataset and produces new trajectories.
                   Multiple runs are parallelized using Spark.
            keep_original: Whether to keep original trajectories in output (default: True).
                          If False, output contains only augmented trajectories.
        """
        super().__init__(times=times, keep_original=keep_original)

    @global_aug
    def apply(self, trajs):
        """Process all trajectories together."""

        if not trajs:
            return []

        # Average all trajectories
        avg_traj = {}
        for key in trajs[0].keys():
            values = [traj[key] for traj in trajs if key in traj]
            values = np.array(values)
            avg_traj[key] = np.mean(values, axis=0)

        return [avg_traj]


# ============================================================================
# Example: Using the pipeline with class-based augmentations
# ============================================================================


def example():
    """Demonstrates the class-based augmentation style."""

    spark = SparkSession.builder.appName("AugmentationExample").getOrCreate()
    pipeline = Pipeline(spark)

    # Add user-defined augmentation classes
    pipeline.add(AddOffsetAugmentation(offset=2.0))
    # Run global augmentation 3 times in parallel using Spark
    # keep_original=True (default): output has 3 (original) + 3 (augmented) = 6 trajectories
    pipeline.add(AverageTrajectoriesAugmentation(times=3))

    # Example with keep_original=False: output has only 3 augmented trajectories
    # pipeline.add(AverageTrajectoriesAugmentation(times=3, keep_original=False))

    # Sample data: list of trajectory dictionaries
    sample_data = [
        {"x": np.array([2, 3, 4]), "y": np.array([5, 6, 7])},
        {"x": np.array([3, 4, 5]), "y": np.array([6, 7, 8])},
        {"x": np.array([4, 5, 6]), "y": np.array([7, 8, 9])},
    ]

    # Run pipeline
    result_rdd = pipeline.run(sample_data)
    results = result_rdd.collect()

    print(f"Input: {len(sample_data)} trajectories")
    print(f"Output: {len(results)} trajectories")
    print("\nResults:")
    for i, traj in enumerate(results):
        print(f"  Trajectory {i}: {traj}")

    spark.stop()


if __name__ == "__main__":
    print("=" * 60)
    print("Example: Class-based augmentation style")
    print("=" * 60)
    example()
