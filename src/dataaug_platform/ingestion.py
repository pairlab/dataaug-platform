import h5py
from pyspark import SparkContext


def load_hdf5_group(group):
    """
    Recursively load an HDF5 group into a nested Python dict.
    - Datasets ? numpy arrays
    - Groups   ? nested dicts
    """
    result = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            result[key] = load_hdf5_group(item)  # recurse
        elif isinstance(item, h5py.Dataset):
            result[key] = item[()]  # read dataset as numpy array
        else:
            # other HDF5 object types (rare)
            result[key] = None
    return result


def hdf5_to_rdd(sc: SparkContext, hdf5_path: str):
    """
    Convert an HDF5 file with structure:
        /data/demo_0, /data/demo_1, ...
    Each demo may contain nested subgroups and datasets.

    Returns:
        RDD[(demo_name, nested_dict)]
    """
    with h5py.File(hdf5_path, "r") as f:
        data_group = f["data"]
        demo_names = list(data_group.keys())

        demo_data_list = []
        for demo_name in demo_names:
            demo_group = data_group[demo_name]
            demo_data = load_hdf5_group(demo_group)
            demo_data_list.append(demo_data)

    return sc.parallelize(demo_data_list)
