# DataAug Platform

A PySpark-based data augmentation framework for trajectory data.

## Installation

```bash
pip install -e .
```

## Usage

```python
from dataaug_platform import Augmentation, local_aug, global_aug, Pipeline
from pyspark.sql import SparkSession

# Define your augmentation
class MyAugmentation(Augmentation):
    @local_aug
    def apply(self, traj):
        # Your augmentation logic
        return modified_traj

# Use in pipeline
spark = SparkSession.builder.appName("MyApp").getOrCreate()
pipeline = Pipeline(spark)
pipeline.add(MyAugmentation())
result = pipeline.run(data)
```

See `examples/` directory for more examples.
