"""
Example demonstrating how to use user-defined augmentation classes.

This shows how to define augmentation classes with @local_aug and @global_aug decorators.
"""

from torch.utils.data import DataLoader

from pyspark.sql import SparkSession
from dataaug_platform import Augmentation, local_aug, global_aug, Pipeline, SparkIterableDataset


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
        import numpy as np
        
        if not trajs:
            return []
        
        # Average all trajectories
        avg_traj = {}
        for key in trajs[0].keys():
            values = [traj[key] for traj in trajs if key in traj]
            if values and isinstance(values[0], np.ndarray):
                avg_traj[key] = np.mean(values, axis=0)
            else:
                avg_traj[key] = values[0] if values else None
        
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
    # keep_original=True (default): output has 5 (original) + 3 (augmented) = 8 trajectories
    pipeline.add(AverageTrajectoriesAugmentation(times=3))
    
    # Example with keep_original=False: output has only 5 augmented trajectories
    # pipeline.add(AverageTrajectoriesAugmentation(times=3, keep_original=False))
    
    # Sample data: list of trajectory dictionaries
    sample_data = [
        {"x": [1, 2, 3], "y": [4, 5, 6]},
        {"x": [2, 3, 4], "y": [5, 6, 7]},
        {"x": [3, 4, 5], "y": [6, 7, 8]},
        {"x": [7, 8, 9], "y": [10, 11, 12]},
        {"x": [11, 12, 13], "y": [14, 15, 16]},
    ]
    
    pipeline.set_data(sample_data)
    
    print("===== Finite Dataset =====")
    spark_dataset = SparkIterableDataset(pipeline)

    spark_dataloader = DataLoader(spark_dataset, batch_size=6, num_workers=1)

    for i, data in enumerate(spark_dataloader):
        print(f'[{i}] {data=}')

    print("===== Infinite Dataset =====")
    num_batches = 4
    batch_count = 0

    print(f"Iteration count {num_batches}")

    spark_dataset = SparkIterableDataset(pipeline, infinite=True)

    spark_dataloader = DataLoader(spark_dataset, batch_size=6, num_workers=1)

    for i, data in enumerate(spark_dataloader):
        if batch_count == num_batches:
            print("Iteration count reached")
            break

        print(f'[{i}] {data=}')
        batch_count += 1
    
    spark.stop()


if __name__ == "__main__":
    print("=" * 60)
    print("Example: Class-based augmentation style")
    print("=" * 60)
    example()
