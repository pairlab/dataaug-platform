import torch
from torch.utils.data import IterableDataset

from .pipeline import Pipeline

class SparkIterableDataset(IterableDataset):
    def __init__(
        self,
        spark_pipeline: Pipeline,
        to_tensor: bool = True,
        infinite: bool = False,
    ):
        super().__init__()
        self.spark_pipeline = spark_pipeline
        self.to_tensor = to_tensor
        self.infinite = infinite

    def _convert(self, item):
        """Convert Spark output item to torch.Tensor/dict-of-tensors if needed."""
        if not self.to_tensor:
            return item

        if isinstance(item, torch.Tensor):
            return item

        if isinstance(item, dict):
            out = {}
            for k, v in item.items():
                if isinstance(v, torch.Tensor):
                    out[k] = v
                else:
                    out[k] = torch.tensor(v, dtype=torch.float32)
            return out

        if isinstance(item, (list, tuple)):
            return torch.tensor(item, dtype=torch.float32)

        if isinstance(item, (int, float)):
            return torch.tensor([item], dtype=torch.float32)

        return item

    def __iter__(self):
        if self.infinite:
            while True:
                rdd = self.spark_pipeline.run()
                for item in rdd.toLocalIterator():
                    yield self._convert(item)
        else:
            rdd = self.spark_pipeline.run()
            for item in rdd.toLocalIterator():
                yield self._convert(item)