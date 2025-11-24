import h5py
from pyspark import SparkContext
from typing import List


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


def read_hdf5_metadata(hdf5_path: str):
    """
    Read all metadata (attributes) from an HDF5 file.
    Returns a nested dictionary containing attributes from the file root
    and all groups/datasets within the file.

    Args:
        hdf5_path: Path to the HDF5 file

    Returns:
        dict: Nested dictionary with structure:
            {
                'root_attrs': {attr_name: attr_value, ...},
                'groups': {
                    group_path: {
                        'attrs': {attr_name: attr_value, ...},
                        'datasets': {
                            dataset_name: {attr_name: attr_value, ...}
                        }
                    }
                }
            }
    """
    metadata = {"root_attrs": {}, "groups": {}}

    with h5py.File(hdf5_path, "r") as f:
        # Read root attributes
        for key in f.attrs.keys():
            metadata["root_attrs"][key] = f.attrs[key]

        # Recursively read group and dataset attributes
        def visit_item(name, obj):
            if isinstance(obj, h5py.Group):
                group_attrs = {}
                for key in obj.attrs.keys():
                    group_attrs[key] = obj.attrs[key]

                metadata["groups"][name] = {"attrs": group_attrs, "datasets": {}}
            elif isinstance(obj, h5py.Dataset):
                dataset_attrs = {}
                for key in obj.attrs.keys():
                    dataset_attrs[key] = obj.attrs[key]

                # Find parent group path
                parent_path = "/".join(name.split("/")[:-1])
                if not parent_path:
                    parent_path = "/"

                if parent_path not in metadata["groups"]:
                    metadata["groups"][parent_path] = {"attrs": {}, "datasets": {}}

                dataset_name = name.split("/")[-1]
                metadata["groups"][parent_path]["datasets"][
                    dataset_name
                ] = dataset_attrs

        f.visititems(visit_item)

    return metadata


def write_trajectories_to_hdf5(trajectories, hdf5_path: str, ignore_keys=None, collate_keys=None):
    """
    Write a list of trajectories to an HDF5 file.
    
    Args:
        trajectories: List of trajectory dictionaries
        hdf5_path: Path to output HDF5 file
        ignore_keys: Optional list of keys to skip when writing
        collate_keys: Optional list of keys whose values are lists/arrays of dicts.
                      These will be collated into a dict of arrays before writing.
    """
    import numpy as np
    
    if ignore_keys is None:
        ignore_keys = []
    if collate_keys is None:
        collate_keys = []
    
    def collate_list_of_dicts(list_of_dicts):
        """Collate a list of dicts into a dict of arrays."""
        if not list_of_dicts:
            return {}
        
        collated = {}
        # Get all unique keys from all dicts
        all_keys = set()
        for d in list_of_dicts:
            if isinstance(d, dict):
                all_keys.update(d.keys())
        
        # Collate each key
        for key in all_keys:
            values = [d.get(key) for d in list_of_dicts if isinstance(d, dict)]
            try:
                collated[key] = np.array(values)
            except (ValueError, TypeError):
                # If can't convert to array, keep as list
                collated[key] = values
        
        return collated
    
    def write_dict_to_group(group, data_dict):
        """Recursively write nested dictionaries to HDF5 groups."""
        for key, value in data_dict.items():
            if key in ignore_keys:
                continue
            
            if isinstance(value, dict):
                nested_grp = group.create_group(key)
                write_dict_to_group(nested_grp, value)
            elif isinstance(value, (list, np.ndarray)):
                # Check if this key should be collated
                if key in collate_keys and isinstance(value, (list, np.ndarray)) and len(value) > 0:
                    if isinstance(value[0], dict):
                        # Collate list of dicts into dict of arrays
                        collated = collate_list_of_dicts(value)
                        nested_grp = group.create_group(key)
                        write_dict_to_group(nested_grp, collated)
                        continue
                
                # Check if list contains dicts or other non-serializable objects
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], dict):
                        # Skip lists of dicts (like observations, datagen_infos)
                        continue
                    # Try to convert list to array
                    try:
                        value = np.array(value)
                    except (ValueError, TypeError):
                        # Skip if can't convert
                        continue
                
                # Write array/list to dataset
                try:
                    group.create_dataset(key, data=value)
                except (TypeError, ValueError) as e:
                    # Skip if data type not supported by HDF5
                    print(f"Warning: Skipping key '{key}' - cannot write to HDF5: {e}")
            elif value is not None and not callable(value):
                try:
                    group.create_dataset(key, data=value)
                except (TypeError, ValueError) as e:
                    print(f"Warning: Skipping key '{key}' - cannot write to HDF5: {e}")
    
    with h5py.File(hdf5_path, "w") as f:
        data_grp = f.create_group("data")
        
        for i, traj in enumerate(trajectories):
            demo_grp = data_grp.create_group(f"demo_{i}")
            write_dict_to_group(demo_grp, traj)
