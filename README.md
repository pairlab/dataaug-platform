# DataAug Platform

A PySpark-based data augmentation framework for trajectory data with integrated MimicGen support.

## Features

- 🚀 **Intuitive API**: Simple, clean pipeline design for data augmentation
- ⚡ **Parallel Processing**: Automatic distribution using PySpark
- 🔧 **Flexible Augmentations**: Easy-to-define local and global augmentations
- 🤖 **MimicGen Integration**: Built-in support for robotic trajectory generation
- 📦 **HDF5 Support**: Seamless data loading and saving

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd dataaug-platform

# Install in editable mode
pip install -e .
```

### Requirements

- Python >= 3.8
- PySpark 4.0.1
- NumPy 2.3.4
- h5py >= 3.0.0
- robosuite 1.4.1
- robomimic 0.5
- mimicgen 1.0.0

## Quick Start

### Basic Usage

```python
from dataaug_platform import Augmentation, local_aug, Pipeline, hdf5_to_rdd
from pyspark.sql import SparkSession
import numpy as np

# Define a custom augmentation
class AddNoiseAugmentation(Augmentation):
    def __init__(self, noise_scale=0.01):
        self.noise_scale = noise_scale

    @local_aug
    def apply(self, traj):
        """Add noise to trajectory states."""
        traj_copy = traj.copy()
        noise = np.random.normal(0, self.noise_scale, traj_copy["states"].shape)
        traj_copy["states"] = traj_copy["states"] + noise
        return traj_copy

# Initialize Spark
spark = SparkSession.builder.appName("DataAugExample").getOrCreate()

# Load data
trajectories = hdf5_to_rdd(spark.sparkContext, "data.hdf5")

# Build and run pipeline
pipeline = Pipeline(spark)
pipeline.add(AddNoiseAugmentation(noise_scale=0.01))
results = pipeline.run(trajectories).collect()
```

### Data Loading and Saving

```python
from dataaug_platform import hdf5_to_rdd, write_trajectories_to_hdf5

# Load HDF5 data
trajectories_rdd = hdf5_to_rdd(spark.sparkContext, "input.hdf5")

# Process with pipeline
results = pipeline.run(trajectories_rdd).collect()

# Save results
write_trajectories_to_hdf5(results, "output.hdf5")
```

## Core Concepts

### 1. Augmentations

Define custom augmentations using decorators:

#### Local Augmentation

Processes each trajectory independently (parallelizable):

```python
class MyLocalAug(Augmentation):
    @local_aug
    def apply(self, traj):
        # Process single trajectory
        return modified_traj
```

#### Global Augmentation

Processes all trajectories together:

```python
class MyGlobalAug(Augmentation):
    @global_aug
    def apply(self, trajs):
        # Process all trajectories
        # Return list of new trajectories
        return new_trajs
```

### 2. Pipeline

Chain augmentations together:

```python
pipeline = Pipeline(spark)
pipeline.add(AugmentationA())
pipeline.add(AugmentationB(times=5, keep_original=True))
results = pipeline.run(data)
```

### 3. Data Ingestion

Simple functions for HDF5 I/O:

- `hdf5_to_rdd(sc, path)` - Load HDF5 to Spark RDD
- `write_trajectories_to_hdf5(trajs, path)` - Write trajectories to HDF5
- `read_hdf5_metadata(path)` - Read HDF5 metadata/attributes
- `load_hdf5_group(group)` - Load HDF5 group recursively

## MimicGen Integration

### Overview

MimicGen generates new robot demonstrations by composing and adapting existing trajectories using subtask-level reasoning.

### Usage

```python
from dataaug_platform import Pipeline, MimicGenAugmentation, hdf5_to_rdd
from mimicgen.configs import MG_TaskSpec
import robomimic.utils.file_utils as FileUtils

# Load data
trajectories_rdd = hdf5_to_rdd(spark.sparkContext, "demos.hdf5")

# Get environment metadata from HDF5
env_meta = FileUtils.get_env_metadata_from_dataset("demos.hdf5")

# Configure task specification
task_spec = MG_TaskSpec()

# Subtask 1: Grasp object
task_spec.add_subtask(
    object_ref="object_name",
    subtask_term_signal="grasp",
    subtask_term_offset_range=(10, 20),
    selection_strategy="nearest_neighbor_object",
    selection_strategy_kwargs={"nn_k": 3},
    action_noise=0.05,
    num_interpolation_steps=5,
)

# Subtask 2: Final placement
task_spec.add_subtask(
    object_ref="target_name",
    subtask_term_signal=None,  # Final subtask
    selection_strategy="random",
    action_noise=0.05,
)

# Build pipeline
pipeline = Pipeline(spark)
pipeline.add(MimicGenAugmentation(
    task_spec=task_spec,
    env_meta=env_meta,  # Pass env metadata
    env_interface_type="MG_TaskName",  # e.g., "MG_Square"
    times=10,  # Generate 10 new trajectories
    keep_original=True,
    num_workers=5,  # Parallel generation
))

# Run pipeline
results = pipeline.run(trajectories_rdd).collect()
```

### Key Design Feature: Per-Worker Environment Initialization

Since robosuite environments cannot be serialized for Spark distribution, the framework uses a per-worker initialization pattern:

1. Pass `env_meta` (serializable dict) instead of `env` object
2. Each Spark worker creates its own environment instance
3. Enables true parallel trajectory generation

This approach:

- ✅ Avoids serialization issues
- ✅ Enables parallel data generation
- ✅ Follows MimicGen's own patterns
- ✅ Clean and maintainable

## Examples

### Basic Example (`examples/example_usage.py`)

Basic example demonstrating pipeline usage with custom augmentations.

```bash
python examples/example_usage.py
```

### MimicGen Square Task (`examples/mimicgen_square_example.py`)

Complete working example with MimicGen for the Square task:

- Proper environment setup with robomimic
- Datagen info preparation
- Full MimicGen data generation pipeline
- Parallel trajectory generation

```bash
python examples/mimicgen_square_example.py \
    --source square_demos.hdf5 \
    --output generated_demos.hdf5 \
    --num-demos 10
```

**Features:**

- Uses `hdf5_to_rdd()` for data loading
- `AddDatagenInfoAugmentation` prepares MimicGen metadata (end-effector poses, object poses, subtask signals)
- Per-worker environment initialization for parallelization
- Achieves 100% success rate in testing

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Your Application                           │
│  • Define augmentations with @local_aug / @global_aug       │
│  • Build pipeline with Pipeline().add()                     │
│  • Load data with hdf5_to_rdd()                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                DataAug Platform                              │
│  • Manages Spark parallelization                            │
│  • Handles data distribution                                │
│  • Coordinates augmentation execution                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                     PySpark                                  │
│  • Distributed computing                                    │
│  • RDD transformations                                      │
│  • Automatic scaling                                        │
└─────────────────────────────────────────────────────────────┘
```

## API Reference

### Core Classes

#### `Augmentation`

Base class for all augmentations.

**Parameters:**

- `times` (int): Number of times to apply augmentation (for global augs)
- `keep_original` (bool): Whether to keep original data

#### `Pipeline`

Manages augmentation workflow.

**Methods:**

- `add(augmentation)`: Add augmentation to pipeline
- `run(data)`: Execute pipeline on data

#### `MimicGenAugmentation`

MimicGen-based trajectory generation.

**Parameters:**

- `task_spec` (MG_TaskSpec): Task specification with subtasks
- `env_meta` (dict): Environment metadata for worker initialization
- `env_interface_type` (str): Environment interface class name
- `times` (int): Number of trajectories to generate
- `keep_original` (bool): Keep source demonstrations
- `num_workers` (int): Number of parallel workers
- `select_src_per_subtask` (bool): Selection strategy flag
- `transform_first_robot_pose` (bool): Transform initial pose
- `interpolate_from_last_target_pose` (bool): Interpolation strategy
- `render` (bool): Enable rendering

### Ingestion Functions

#### `hdf5_to_rdd(sc, hdf5_path)`

Load HDF5 file with structure `/data/demo_0`, `/data/demo_1`, etc. into Spark RDD.

#### `write_trajectories_to_hdf5(trajectories, hdf5_path)`

Write list of trajectory dictionaries to HDF5 file.

#### `read_hdf5_metadata(hdf5_path)`

Read all metadata/attributes from HDF5 file.

#### `load_hdf5_group(group)`

Recursively load HDF5 group into nested Python dict.

## Tips and Best Practices

1. **Start Simple**: Test augmentations locally before scaling to Spark
2. **Monitor Memory**: Use `spark.driver.memory` config for large datasets
3. **Use Partitioning**: Adjust `num_workers` based on cluster size
4. **Check Spark UI**: Monitor job progress at http://localhost:4040
5. **Test Incrementally**: Validate each augmentation before chaining

## Troubleshooting

### Issue: Out of memory errors

**Solution**: Increase Spark driver memory:

```python
spark = SparkSession.builder \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()
```

### Issue: Slow execution

**Solution**: Adjust parallelization:

```python
pipeline.add(MimicGenAugmentation(..., num_workers=10))
```

### Issue: Environment serialization errors

**Solution**: Use `env_meta` pattern as shown in MimicGen examples

## Contributing

To add new augmentations:

1. Inherit from `Augmentation`
2. Use `@local_aug` or `@global_aug` decorator
3. Implement `apply()` method
4. Test locally before scaling

Example:

```python
class MyAugmentation(Augmentation):
    @local_aug
    def apply(self, traj):
        # Your logic here
        return modified_traj
```
