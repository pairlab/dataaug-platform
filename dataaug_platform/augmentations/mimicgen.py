from .base_augmentation import Augmentation, global_aug
from mimicgen.datagen.waypoint import WaypointSequence, WaypointTrajectory
import mimicgen.utils.pose_utils as PoseUtils
from mimicgen.datagen.datagen_info import DatagenInfo
from mimicgen.datagen.selection_strategy import make_selection_strategy
from mimicgen.configs import MG_TaskSpec
import numpy as np


class MimicGenAugmentation(Augmentation):
    """
    Global augmentation that applies MimicGen data generation to create new trajectories.

    This augmentation wraps the MimicGen DataGenerator and integrates it into the dataaug
    platform. It processes the entire dataset at once to generate new trajectories by
    transforming and recombining source demonstrations.

    Example:
        from mimicgen.configs import MG_TaskSpec

        # Create task specification
        task_spec = MG_TaskSpec()
        task_spec.add_subtask(
            object_ref="cube",
            subtask_term_signal="grasp",
            subtask_term_offset_range=(-10, 10),
            selection_strategy="nearest_neighbor_object",
            num_interpolation_steps=5,
        )
        task_spec.add_subtask(
            object_ref="bin",
            subtask_term_signal=None,  # final subtask
            selection_strategy="nearest_neighbor_object",
        )

        # Create augmentation
        aug = MimicGenAugmentation(
            task_spec=task_spec,
            env_meta=env_meta,  # Pass env metadata, not env itself
            env_interface_type="MG_Square",  # Interface type
            times=10,  # generate 10 new trajectories
            keep_original=True
        )
    """

    def __init__(
        self,
        task_spec,
        env_meta,
        env_interface_type,
        times=1,
        keep_original=True,
        num_workers=5,
        select_src_per_subtask=False,
        transform_first_robot_pose=False,
        interpolate_from_last_target_pose=True,
        render=False,
        video_writer=None,
        video_skip=5,
        camera_names=None,
    ):
        """
        Initialize the MimicGen augmentation.

        Args:
            task_spec (MG_TaskSpec): Task specification for MimicGen data generation
            env_meta (dict): Environment metadata for creating environment in each worker
            env_interface_type (str): Type of environment interface (e.g., "MG_Square")
            times (int): Number of new trajectories to generate. Default is 1.
            keep_original (bool): Whether to keep original trajectories. Default is True.
            num_workers (int): Number of parallel workers for generation
            select_src_per_subtask (bool): If True, select different source demo per subtask
            transform_first_robot_pose (bool): If True, include first robot pose in transformation
            interpolate_from_last_target_pose (bool): If True, interpolate from last target pose
            render (bool): If True, render during generation
            video_writer: Optional video writer for recording
            video_skip (int): Frame skip rate for video recording
            camera_names (list): Camera names for rendering
        """
        super().__init__(
            times=times, keep_original=keep_original, num_workers=num_workers
        )

        assert isinstance(
            task_spec, MG_TaskSpec
        ), "task_spec must be an MG_TaskSpec instance"

        self.task_spec = task_spec
        self.env_meta = env_meta
        self.env_interface_type = env_interface_type
        self.select_src_per_subtask = select_src_per_subtask
        self.transform_first_robot_pose = transform_first_robot_pose
        self.interpolate_from_last_target_pose = interpolate_from_last_target_pose
        self.render = render
        self.video_writer = video_writer
        self.video_skip = video_skip
        self.camera_names = camera_names

        # Environment will be initialized in each worker
        self.env = None
        self.env_interface = None

    def _initialize_data_generator(self, trajs):
        """
        Initialize the DataGenerator from trajectory data.

        This method converts the trajectory list into the format expected by DataGenerator
        by creating a temporary dataset or by directly loading the dataset info.
        """

        # Parse trajectories to extract datagen info
        src_dataset_infos = []
        src_subtask_indices = []
        subtask_names = []

        for traj_idx, traj in enumerate(trajs):
            # Extract datagen_info from trajectory
            datagen_info = traj.get("datagen_info", {})

            # Create DatagenInfo object
            datagen_info_obj = DatagenInfo(
                eef_pose=datagen_info.get("eef_pose"),
                object_poses=datagen_info.get("object_poses", {}),
                subtask_term_signals=datagen_info.get("subtask_term_signals", {}),
                target_pose=datagen_info.get("target_pose"),
                gripper_action=datagen_info.get("gripper_action"),
            )
            src_dataset_infos.append(datagen_info_obj)

            # Parse subtask indices using task spec
            ep_subtask_indices = []
            prev_subtask_term_ind = 0

            for subtask_ind in range(len(self.task_spec)):
                subtask_term_signal = self.task_spec[subtask_ind]["subtask_term_signal"]

                if subtask_term_signal is None:
                    # Final subtask ends at trajectory end
                    subtask_term_ind = len(traj.get("actions", []))
                else:
                    # Detect 0 -> 1 transition in subtask signal
                    subtask_indicators = datagen_info_obj.subtask_term_signals[
                        subtask_term_signal
                    ]
                    diffs = subtask_indicators[1:] - subtask_indicators[:-1]
                    # Find first index where subtask completes
                    end_ind = next(
                        (
                            int(nz) + 1
                            for nz in diffs.nonzero()[0]
                            if int(nz) + 1 >= prev_subtask_term_ind
                        ),
                        None,
                    )
                    if end_ind is None:
                        raise ValueError(
                            f"No valid subtask termination found for '{subtask_term_signal}' "
                            f"in trajectory {traj_idx}"
                        )
                    subtask_term_ind = end_ind + 1

                ep_subtask_indices.append([prev_subtask_term_ind, subtask_term_ind])
                prev_subtask_term_ind = subtask_term_ind

                # Collect subtask names on first trajectory
                if traj_idx == 0:
                    subtask_names.append(subtask_term_signal)

            src_subtask_indices.append(ep_subtask_indices)

        # Store parsed data
        self.src_dataset_infos = src_dataset_infos
        self.src_subtask_indices = src_subtask_indices
        self.subtask_names = subtask_names
        self.demo_keys = [f"demo_{i}" for i in range(len(trajs))]

    def _initialize_environment(self):
        """
        Initialize environment and interface in the worker.
        This is called once per worker to create local environment instances.
        """
        if self.env is None:
            import robomimic.utils.env_utils as EnvUtils
            from mimicgen.env_interfaces.base import make_interface
            
            # Create environment using robomimic EnvUtils
            self.env = EnvUtils.create_env_for_data_processing(
                env_meta=self.env_meta,
                camera_names=self.camera_names or [],
                camera_height=84,
                camera_width=84,
                reward_shaping=False,
                render=self.render,
                render_offscreen=False,
                use_image_obs=False,
                use_depth_obs=False,
            )
            
            # Create environment interface
            self.env_interface = make_interface(
                name=self.env_interface_type,
                interface_type="robosuite",
                env=self.env.base_env,  # Interface needs the base robosuite env
            )

    @global_aug
    def apply(self, trajs):
        """
        Apply MimicGen data generation to create new trajectories.

        Args:
            trajs (list): List of trajectory dictionaries. Each trajectory should contain:
                - "datagen_info": dict with keys:
                    - "eef_pose": end-effector poses
                    - "object_poses": dict of object poses
                    - "subtask_term_signals": dict of subtask termination signals
                    - "target_pose": target poses
                    - "gripper_action": gripper actions
                - "actions": action array

        Returns:
            list: List of newly generated trajectory dictionaries
        """
        if len(trajs) == 0:
            return []

        # Initialize environment in this worker (if not already initialized)
        self._initialize_environment()

        # Initialize data generator with source trajectories
        self._initialize_data_generator(trajs)

        # Generate a new trajectory using MimicGen
        result = self._generate_trajectory()

        # Convert result to trajectory dictionary format
        new_traj = {
            "initial_state": result["initial_state"],
            "states": result["states"],
            "obs": result["observations"],
            "datagen_info": result["datagen_infos"],
            "actions": result["actions"],
            "success": result["success"],
            "src_demo_inds": result["src_demo_inds"],
            "src_demo_labels": result["src_demo_labels"],
        }

        return [new_traj]

    def _generate_trajectory(self):
        """
        Generate a single trajectory using MimicGen algorithm.

        This method implements the core MimicGen generation logic, adapted from
        DataGenerator.generate() to work with the parsed trajectory data.
        """

        # Sample new task instance
        self.env.reset()
        new_initial_state = np.array(self.env.base_env.sim.get_state().flatten())

        # Sample new subtask boundaries
        all_subtask_inds = self._randomize_subtask_boundaries()

        # State variables for generation
        selected_src_demo_ind = None
        prev_executed_traj = None

        # Storage for generated data
        generated_states = []
        generated_obs = []
        generated_datagen_infos = []
        generated_actions = []
        generated_success = False
        generated_src_demo_inds = []
        generated_src_demo_labels = []

        for subtask_ind in range(len(self.task_spec)):
            is_first_subtask = subtask_ind == 0

            # Get current datagen info
            cur_datagen_info = self.env_interface.get_datagen_info()

            subtask_object_name = self.task_spec[subtask_ind]["object_ref"]
            cur_object_pose = (
                cur_datagen_info.object_poses[subtask_object_name]
                if subtask_object_name is not None
                else None
            )

            # Source demo selection
            need_source_demo_selection = is_first_subtask or self.select_src_per_subtask

            if need_source_demo_selection:
                selected_src_demo_ind = self._select_source_demo(
                    eef_pose=cur_datagen_info.eef_pose,
                    object_pose=cur_object_pose,
                    subtask_ind=subtask_ind,
                    src_subtask_inds=all_subtask_inds[:, subtask_ind],
                    subtask_object_name=subtask_object_name,
                )

            # Get source subtask segment
            selected_src_subtask_inds = all_subtask_inds[
                selected_src_demo_ind, subtask_ind
            ]
            src_ep_datagen_info = self.src_dataset_infos[selected_src_demo_ind]

            src_subtask_eef_poses = src_ep_datagen_info.eef_pose[
                selected_src_subtask_inds[0] : selected_src_subtask_inds[1]
            ]
            src_subtask_target_poses = src_ep_datagen_info.target_pose[
                selected_src_subtask_inds[0] : selected_src_subtask_inds[1]
            ]
            src_subtask_gripper_actions = src_ep_datagen_info.gripper_action[
                selected_src_subtask_inds[0] : selected_src_subtask_inds[1]
            ]

            src_subtask_object_pose = (
                src_ep_datagen_info.object_poses[subtask_object_name][
                    selected_src_subtask_inds[0]
                ]
                if subtask_object_name is not None
                else None
            )

            # Prepare source poses
            if is_first_subtask or self.transform_first_robot_pose:
                src_eef_poses = np.concatenate(
                    [src_subtask_eef_poses[0:1], src_subtask_target_poses], axis=0
                )
            else:
                src_eef_poses = src_subtask_target_poses

            # Duplicate first gripper action to match pose array length  
            src_subtask_gripper_actions = np.concatenate(
                [src_subtask_gripper_actions[[0]], src_subtask_gripper_actions], axis=0
            )

            # Transform source segment
            if subtask_object_name is not None:
                transformed_eef_poses = (
                    PoseUtils.transform_source_data_segment_using_object_pose(
                        obj_pose=cur_object_pose,
                        src_eef_poses=src_eef_poses,
                        src_obj_pose=src_subtask_object_pose,
                    )
                )
            else:
                transformed_eef_poses = src_eef_poses

            # Build trajectory to execute
            traj_to_execute = WaypointTrajectory()

            if self.interpolate_from_last_target_pose and (not is_first_subtask):
                last_waypoint = prev_executed_traj.last_waypoint
                init_sequence = WaypointSequence(sequence=[last_waypoint])
            else:
                init_sequence = WaypointSequence.from_poses(
                    poses=cur_datagen_info.eef_pose[None],
                    gripper_actions=src_subtask_gripper_actions[0:1],
                    action_noise=self.task_spec[subtask_ind]["action_noise"],
                )
            traj_to_execute.add_waypoint_sequence(init_sequence)

            # Add transformed segment
            transformed_seq = WaypointSequence.from_poses(
                poses=transformed_eef_poses,
                gripper_actions=src_subtask_gripper_actions,
                action_noise=self.task_spec[subtask_ind]["action_noise"],
            )
            transformed_traj = WaypointTrajectory()
            transformed_traj.add_waypoint_sequence(transformed_seq)

            # Merge with interpolation
            traj_to_execute.merge(
                transformed_traj,
                num_steps_interp=self.task_spec[subtask_ind]["num_interpolation_steps"],
                num_steps_fixed=self.task_spec[subtask_ind]["num_fixed_steps"],
                action_noise=(
                    float(
                        self.task_spec[subtask_ind]["apply_noise_during_interpolation"]
                    )
                    * self.task_spec[subtask_ind]["action_noise"]
                ),
            )

            traj_to_execute.pop_first()

            # Execute trajectory
            exec_results = traj_to_execute.execute(
                env=self.env,
                env_interface=self.env_interface,
                render=self.render,
                video_writer=self.video_writer,
                video_skip=self.video_skip,
                camera_names=self.camera_names,
            )

            # Collect results
            if len(exec_results["states"]) > 0:
                generated_states += exec_results["states"]
                generated_obs += exec_results["observations"]
                generated_datagen_infos += exec_results["datagen_infos"]
                generated_actions.append(exec_results["actions"])
                generated_success = generated_success or exec_results["success"]
                generated_src_demo_inds.append(selected_src_demo_ind)
                generated_src_demo_labels.append(
                    selected_src_demo_ind
                    * np.ones((exec_results["actions"].shape[0], 1), dtype=int)
                )

            prev_executed_traj = traj_to_execute

        # Merge arrays
        if len(generated_actions) > 0:
            generated_actions = np.concatenate(generated_actions, axis=0)
            generated_src_demo_labels = np.concatenate(
                generated_src_demo_labels, axis=0
            )
        
        # Convert lists to numpy arrays (avoids object dtype)
        if len(generated_states) > 0:
            generated_states = np.array(generated_states)
        else:
            generated_states = np.array([])
        
        if len(generated_src_demo_inds) > 0:
            generated_src_demo_inds = np.array(generated_src_demo_inds, dtype=np.int32)
        else:
            generated_src_demo_inds = np.array([], dtype=np.int32)

        return dict(
            initial_state=new_initial_state,
            states=generated_states,
            observations=generated_obs,
            datagen_infos=generated_datagen_infos,
            actions=generated_actions,
            success=generated_success,
            src_demo_inds=generated_src_demo_inds,
            src_demo_labels=generated_src_demo_labels,
        )

    def _randomize_subtask_boundaries(self):
        """Apply random offsets to subtask boundaries according to task spec."""
        src_subtask_indices = np.array(self.src_subtask_indices)

        for i in range(src_subtask_indices.shape[1] - 1):
            end_offsets = np.random.randint(
                low=self.task_spec[i]["subtask_term_offset_range"][0],
                high=self.task_spec[i]["subtask_term_offset_range"][1] + 1,
                size=src_subtask_indices.shape[0],
            )
            src_subtask_indices[:, i, 1] = src_subtask_indices[:, i, 1] + end_offsets
            src_subtask_indices[:, i + 1, 0] = src_subtask_indices[:, i, 1]

        return src_subtask_indices

    def _select_source_demo(
        self, eef_pose, object_pose, subtask_ind, src_subtask_inds, subtask_object_name
    ):
        """Select source demonstration for current subtask."""

        if subtask_object_name is None:
            assert self.task_spec[subtask_ind]["selection_strategy"] == "random"

        # Collect datagen info for subtask segments
        src_subtask_datagen_infos = []
        for i in range(len(self.demo_keys)):
            src_ep_datagen_info = self.src_dataset_infos[i]
            subtask_start_ind = src_subtask_inds[i][0]
            subtask_end_ind = src_subtask_inds[i][1]

            src_subtask_datagen_infos.append(
                DatagenInfo(
                    eef_pose=src_ep_datagen_info.eef_pose[
                        subtask_start_ind:subtask_end_ind
                    ],
                    object_poses=(
                        {
                            subtask_object_name: src_ep_datagen_info.object_poses[
                                subtask_object_name
                            ][subtask_start_ind:subtask_end_ind]
                        }
                        if subtask_object_name is not None
                        else None
                    ),
                    subtask_term_signals=None,
                    target_pose=src_ep_datagen_info.target_pose[
                        subtask_start_ind:subtask_end_ind
                    ],
                    gripper_action=src_ep_datagen_info.gripper_action[
                        subtask_start_ind:subtask_end_ind
                    ],
                )
            )

        # Run selection
        selection_strategy_obj = make_selection_strategy(
            self.task_spec[subtask_ind]["selection_strategy"]
        )
        selection_strategy_kwargs = (
            self.task_spec[subtask_ind]["selection_strategy_kwargs"] or {}
        )

        selected_src_demo_ind = selection_strategy_obj.select_source_demo(
            eef_pose=eef_pose,
            object_pose=object_pose,
            src_subtask_datagen_infos=src_subtask_datagen_infos,
            **selection_strategy_kwargs,
        )

        return selected_src_demo_ind
