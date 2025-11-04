from abc import ABC, abstractmethod
from pyspark.sql import SparkSession

# -----------------------
# Decorators
# -----------------------


def local_aug(func):
    """
    Decorator for local augmentation methods.
    Handles conversion from RDD to individual trajectory dicts and back.

    Usage:
        class MyAugmentation(Augmentation):
            @local_aug
            def apply(self, traj):
                # traj is a dict representing one trajectory
                # Return a modified dict
                return modified_traj
    """
    func._aug_type = "local"
    return func


def global_aug(func):
    """
    Decorator for global augmentation methods.
    Handles conversion from RDD to list of trajectory dicts and back.
    
    Global augmentations process the entire dataset at once. They can be run
    multiple times by passing `times > 1` to the `__init__` method. Multiple
    runs are parallelized across Spark workers for efficiency.

    Usage:
        class MyAugmentation(Augmentation):
            def __init__(self, times=1, keep_original=True):
                super().__init__(times=times, keep_original=keep_original)
            
            @global_aug
            def apply(self, trajs):
                # trajs is a list of dicts representing all trajectories
                # Return a list of new trajectory dicts
                return new_trajs
    """
    func._aug_type = "global"
    return func


# -----------------------
# Base Augmentation Types
# -----------------------


class Augmentation(ABC):
    """
    Abstract base class for all augmentations.

    Users can define augmentation classes that inherit from this base class.
    The apply method should be decorated with either @local_aug or @global_aug.

    Example:
        class MyLocalAug(Augmentation):
            def __init__(self, param):
                self.param = param

            @local_aug
            def apply(self, traj):
                # Modify individual trajectory
                return modified_traj

        class MyGlobalAug(Augmentation):
            def __init__(self, times=1, keep_original=True):
                super().__init__(times=times, keep_original=keep_original)
            
            @global_aug
            def apply(self, trajs):
                # Process all trajectories
                return new_trajs
    """

    def __init__(self, times=1, keep_original=True):
        """
        Initialize the augmentation.
        
        Args:
            times: For global augmentations, number of times to run the augmentation.
                   Each run processes the whole dataset and produces trajectories.
                   Default is 1. Local augmentations ignore this parameter.
                   Must be >= 1.
            keep_original: For global augmentations, whether to keep the original trajectories
                          in the output. If True, output contains original + augmented trajectories.
                          If False, output contains only augmented trajectories.
                          Default is True. Local augmentations ignore this parameter.
        
        Raises:
            ValueError: If times is less than 1.
        """
        if times < 1:
            raise ValueError("times must be >= 1")
        self.times = times
        self.keep_original = keep_original

    def _apply_rdd(self, rdd):
        """
        Internal method that handles RDD conversion and calls the decorated apply method.
        This is called by the Pipeline.
        """

        # Check if apply method has a decorator
        if not hasattr(self.apply, "_aug_type"):
            raise ValueError(
                "apply method must be decorated with either @local_aug or @global_aug"
            )

        aug_type = self.apply._aug_type

        if aug_type == "local":
            # Local augmentation: map over each trajectory
            return rdd.map(lambda traj: self.apply(traj))

        elif aug_type == "global":
            # Global augmentation: collect all trajectories, apply, return new RDD
            sc = rdd.context
            all_trajs = rdd.collect()  # materialize all trajectories
            
            # Broadcast the full dataset to all workers
            broadcast_data = sc.broadcast(all_trajs)
            
            # Run the global augmentation multiple times using Spark parallelism
            # Create an RDD with range(self.times) and map each to run the augmentation
            def run_augmentation_once(_):
                """Run the global augmentation once with the broadcast data."""
                result = self.apply(broadcast_data.value)
                # Ensure result is a list
                if not isinstance(result, list):
                    result = [result]
                return result
            
            # Parallelize the multiple runs across Spark workers
            runs_rdd = sc.parallelize(range(self.times), numSlices=self.times)
            new_trajs_nested = runs_rdd.map(run_augmentation_once).collect()
            
            # Flatten the nested list: [[traj1, traj2], [traj3, traj4]] -> [traj1, traj2, traj3, traj4]
            new_trajs = [traj for run_result in new_trajs_nested for traj in run_result]
            
            # Convert new trajectories to RDD
            new_trajs_rdd = sc.parallelize(new_trajs)
            
            # Conditionally union original with new trajectories based on keep_original flag
            if self.keep_original:
                return rdd.union(new_trajs_rdd)
            else:
                return new_trajs_rdd

        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")

    @abstractmethod
    def apply(self, *args, **kwargs):
        """
        Apply the augmentation. Must be decorated with @local_aug or @global_aug.

        For local aug: apply(self, traj: dict) -> dict
        For global aug: apply(self, trajs: list[dict]) -> list[dict]
        """
        raise NotImplementedError("Augmentation class must implement the apply method.")
