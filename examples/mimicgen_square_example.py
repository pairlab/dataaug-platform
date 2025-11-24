"""
MimicGen Square Task Example

This example demonstrates how to use the MimicGenAugmentation with your data pipeline.
It shows how intuitive it is to:
1. Load HDF5 trajectory data using ingestion functions
2. Prepare trajectories with datagen_info using a local augmentation
3. Apply MimicGen data generation using the pipeline
4. Generate new trajectories for the Square task

The Square task involves:
- Subtask 1: Grasp the square nut
- Subtask 2: Place it on the square peg
"""

import numpy as np
from pyspark.sql import SparkSession
from dataaug_platform import (
    Pipeline,
    Augmentation,
    local_aug,
    MimicGenAugmentation,
    hdf5_to_rdd,
    write_trajectories_to_hdf5,
)
from mimicgen.configs import MG_TaskSpec
from mimicgen.env_interfaces.robosuite import MG_Square
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils


# ============================================================================
# Step 1: Define a local augmentation to add datagen_info to trajectories
# ============================================================================


class AddDatagenInfoAugmentation(Augmentation):
    """
    Local augmentation that adds datagen_info to each trajectory.

    This is necessary because MimicGen requires datagen_info containing:
    - eef_pose: end-effector poses
    - object_poses: dictionary of object poses
    - subtask_term_signals: dictionary of subtask termination signals
    - target_pose: target poses
    - gripper_action: gripper actions
    """

    def __init__(self, robomimic_env, base_env, env_interface):
        """
        Args:
            robomimic_env: Robomimic environment wrapper
            base_env: Base robosuite environment instance
            env_interface: MimicGen environment interface
        """
        self.robomimic_env = robomimic_env
        self.base_env = base_env
        self.env_interface = env_interface

    @local_aug
    def apply(self, traj):
        """
        Add datagen_info to a single trajectory by replaying it.

        Args:
            traj: Trajectory dictionary with 'states' and 'actions'

        Returns:
            Trajectory dictionary with added 'datagen_info'
        """
        # Extract datagen info by replaying the trajectory
        datagen_infos = []
        target_poses = []
        gripper_actions = []

        # Reset environment
        self.robomimic_env.reset()

        # Replay trajectory and collect datagen info
        for i, (state, action) in enumerate(zip(traj["states"], traj["actions"])):
            # Set sim state on base robosuite environment
            self.base_env.sim.set_state_from_flattened(state)
            self.base_env.sim.forward()

            # Get datagen info at this state
            datagen_info = self.env_interface.get_datagen_info()
            datagen_infos.append(datagen_info)

            # Extract target pose and gripper action from the action
            target_pose = self.env_interface.action_to_target_pose(action[:-1])
            target_poses.append(target_pose)
            gripper_action = self.env_interface.action_to_gripper_action(action)
            gripper_actions.append(gripper_action)

        # Convert list of DatagenInfo objects to dictionary format
        datagen_info_dict = {
            "eef_pose": np.array([di.eef_pose for di in datagen_infos]),
            "object_poses": {},
            "subtask_term_signals": {},
            "target_pose": np.array(target_poses),
            "gripper_action": np.array(gripper_actions),
        }

        # Collect object poses
        if datagen_infos:
            for obj_name in datagen_infos[0].object_poses.keys():
                datagen_info_dict["object_poses"][obj_name] = np.array(
                    [di.object_poses[obj_name] for di in datagen_infos]
                )

            # Collect subtask termination signals
            for signal_name in datagen_infos[0].subtask_term_signals.keys():
                datagen_info_dict["subtask_term_signals"][signal_name] = np.array(
                    [di.subtask_term_signals[signal_name] for di in datagen_infos]
                )

        # Add datagen_info to trajectory
        traj_with_info = traj.copy()
        traj_with_info["datagen_info"] = datagen_info_dict

        return traj_with_info


# ============================================================================
# Step 2: Main example function
# ============================================================================


def run_mimicgen_square_example(
    source_hdf5_path,
    output_hdf5_path=None,
    num_demos_to_generate=5,
):
    """
    Run the MimicGen Square task example.

    Args:
        source_hdf5_path: Path to source demonstration HDF5 file
        output_hdf5_path: Optional path to save generated demonstrations
        num_demos_to_generate: Number of new demonstrations to generate
    """

    print("=" * 80)
    print("MimicGen Square Task Example")
    print("=" * 80)

    # Initialize Spark
    spark = (
        SparkSession.builder.appName("MimicGenSquareExample")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

    print("\n[1/6] Loading source demonstrations from HDF5...")
    # Load trajectories using ingestion function
    trajectories_rdd = hdf5_to_rdd(spark.sparkContext, source_hdf5_path)
    num_source_demos = trajectories_rdd.count()
    print(f"       Loaded {num_source_demos} source demonstrations")

    # Initialize environment for Square task using robomimic EnvUtils
    print("\n[2/6] Initializing Square task environment...")
    # Read environment metadata from the HDF5 file
    env_meta = FileUtils.get_env_metadata_from_dataset(source_hdf5_path)

    # Create environment using robomimic EnvUtils
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=[],
        camera_height=84,
        camera_width=84,
        reward_shaping=False,
    )
    # MimicGen interfaces expect the base robosuite env, not the robomimic wrapper
    env_interface = MG_Square(env=env.base_env)
    print("       Environment initialized successfully")

    # Create task specification for Square task
    print("\n[3/6] Creating MimicGen task specification...")
    task_spec = MG_TaskSpec()

    # Subtask 1: Grasp the square nut
    task_spec.add_subtask(
        object_ref="square_nut",
        subtask_term_signal="grasp",
        subtask_term_offset_range=(10, 20),
        selection_strategy="nearest_neighbor_object",
        selection_strategy_kwargs={"nn_k": 3},
        action_noise=0.05,
        num_interpolation_steps=5,
        num_fixed_steps=0,
        apply_noise_during_interpolation=False,
    )

    # Subtask 2: Place on square peg (final subtask)
    task_spec.add_subtask(
        object_ref="square_peg",
        subtask_term_signal=None,  # Final subtask
        subtask_term_offset_range=None,
        selection_strategy="random",
        selection_strategy_kwargs=None,
        action_noise=0.05,
        num_interpolation_steps=5,
        num_fixed_steps=0,
        apply_noise_during_interpolation=False,
    )
    print("       Task spec configured with 2 subtasks")

    # Step 1: Add datagen_info to trajectories
    # This step requires environment access, so we do it locally before the pipeline.
    # We use the locally created environment since we only need to process once.
    print("\n[4/6] Adding datagen_info to source demonstrations...")
    print("       (Processing locally - only needs to be done once)")
    trajectories = trajectories_rdd.collect()
    aug = AddDatagenInfoAugmentation(env, env.base_env, env_interface)
    trajectories_with_info = [aug.apply(traj) for traj in trajectories]
    print(f"       Processed {len(trajectories_with_info)} trajectories")

    # Step 2: Build and run pipeline with MimicGen augmentation
    print("\n[5/6] Building data augmentation pipeline...")
    pipeline = Pipeline(spark)

    # Add MimicGen augmentation to pipeline
    # NOTE: We pass env_meta (serializable) instead of env (not serializable)
    # Each Spark worker will initialize its own local environment
    print(f"       - Adding MimicGenAugmentation (global, {num_demos_to_generate}x)")
    pipeline.add(
        MimicGenAugmentation(
            task_spec=task_spec,
            env_meta=env_meta,  # Pass metadata, not the env itself
            env_interface_type="MG_Square",  # Interface type to create in worker
            times=num_demos_to_generate,
            keep_original=True,
            num_workers=3,  # Can now parallelize!
            select_src_per_subtask=False,
            transform_first_robot_pose=False,
            interpolate_from_last_target_pose=True,
            render=False,
        )
    )
    print("       Pipeline configured successfully")

    # Run pipeline - this is where the magic happens!
    print("\n[6/6] Running data augmentation pipeline...")
    print(f"       Generating {num_demos_to_generate} new demonstrations...")
    result_rdd = pipeline.run(trajectories_with_info)

    # Collect results
    print("       Collecting results...")
    results = result_rdd.collect()
    print(f"       Total trajectories: {len(results)}")
    print(f"       - Source demos: {num_source_demos}")
    print(f"       - Generated demos: {len(results) - num_source_demos}")

    # Display statistics
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)

    num_generated = len(results) - num_source_demos
    successful_generated = sum(
        1
        for i, traj in enumerate(results[num_source_demos:])
        if traj.get("success", False)
    )
    print(f"Successful generations: {successful_generated}/{num_generated}")

    if len(results) > num_source_demos:
        print("\nSample generated trajectory info:")
        sample_traj = results[num_source_demos]
        print(f"  - States: {len(sample_traj['states'])}")
        print(f"  - Actions shape: {sample_traj['actions'].shape}")
        print(f"  - Success: {sample_traj.get('success', False)}")
        print(f"  - Source demo indices: {sample_traj.get('src_demo_inds', [])}")

    # Save results if output path provided
    if output_hdf5_path:
        print(f"\n[Optional] Saving results to {output_hdf5_path}...")
        write_trajectories_to_hdf5(
            results, output_hdf5_path, ignore_keys=["datagen_info"]
        )
        print("       Results saved successfully")

    spark.stop()
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)

    return results


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run MimicGen Square task example with data augmentation pipeline"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to source demonstration HDF5 file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save generated demonstrations (optional)",
    )
    parser.add_argument(
        "--num-demos",
        type=int,
        default=5,
        help="Number of demonstrations to generate (default: 5)",
    )

    args = parser.parse_args()

    # Run example
    results = run_mimicgen_square_example(
        source_hdf5_path=args.source,
        output_hdf5_path=args.output,
        num_demos_to_generate=args.num_demos,
    )

    print(f"\nGenerated {len(results)} total trajectories")
    print("Use these trajectories for robot learning!")
