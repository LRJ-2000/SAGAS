import mujoco
import numpy as np
import colorsys
from dm_control import mjcf

from ogbenchTL.manipspace import lie
from ogbenchTL.manipspace.envs.manipspace_env import ManipSpaceEnv


class TLManipEnv(ManipSpaceEnv):
    """Temporal Logic environment for manipulation.

    This environment consists of multiple regions. The goal is to reach regions according to a temporal logic task.
    """

    def __init__(
        self,
        region_sample_seed=0,
        region_num=0,
        region_type="sphere",
        nonoverlapping_regions=True,
        sample_regions_func=None,
        sample_regions_params=None,
        override_regions=None,
        *args,
        **kwargs,
    ):
        """
        Initialize the Temporal Logic environment.

        Args:
            region_sample_seed: Random seed for region sampling if using the default sampling method.
            region_num: Number of regions to sample in the environment.
            region_type: Type of regions to sample. Currently only "sphere" is supported.
            nonoverlapping_regions: Whether to ensure sampled regions do not overlap.
            sample_regions_func: A function to sample regions dynamically, needs to be of the form:
                - sample_regions(region_num) -> List of regions, each region is (x, y, z, r, rgba)
            sample_regions_params: Parameters for the sample_regions_func, supported keys:
                - safe_margin: Margin from workspace bounds to sample regions. default: 0.1.
                - r_range: Range of region radii respecting to the percentage of workspace size. default: (0.02, 0.05).
                - alpha: Alpha value for region colors. default: 0.6.
            override_regions: If provided, directly use these regions instead of sampling.
            *args: Additional arguments for the ManipSpaceEnv.
            **kwargs: Additional keyword arguments for the ManipSpaceEnv.
        """
        assert region_num >= 0, "region_num must be greater than or equal to 0."

        self._region_sample_seed = region_sample_seed
        self._region_num = region_num
        self._region_type = region_type
        self._nonoverlapping_regions = nonoverlapping_regions
        self._override_regions = override_regions

        if override_regions is not None:
            self._regions = override_regions
        else:
            self._regions = None

        # Dynamic replacement of region sampling function.
        if sample_regions_func is not None:
            self._sample_regions = lambda self, region_num: sample_regions_func(
                region_num
            )

        if sample_regions_params is not None:
            self._sample_regions_params = sample_regions_params
        else:
            self._sample_regions_params = {
                "safe_margin": 0.1,
                "r_range": (0.05, 0.2),
                "alpha": 0.6,
            }

        super().__init__(*args, **kwargs)

    def _generate_colors(self, n, s_range=(0.7, 1.0), v_range=(0.7, 1.0)):
        golden_ratio_conjugate = (1 + np.sqrt(5)) / 2 - 1
        colors = []
        for i in range(n):
            hue = (i * golden_ratio_conjugate) % 1
            saturation = np.random.uniform(s_range[0], s_range[1])
            value = np.random.uniform(v_range[0], v_range[1])
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
        return colors

    def set_regions(self, regions):
        """Set the regions of the environment.

        Args:
            regions: List of regions, each region is (x, y, z, r, rgba)
        """
        self._regions_obs = regions

        # Convert regions from observation space to environment space
        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0
        
        env_regions = []
        for region in regions:
            x, y, z, r, rgba = region
            # Apply inverse transformation: pos_env = pos_obs / scaler + center
            # Note: z in obs space is 0-3, z in env space is ~0-0.3. 
            # The scaler 10.0 works for this.
            
            pos_obs = np.array([x, y, z])
            pos_env = pos_obs / xyz_scaler + xyz_center
            r_env = r / xyz_scaler
            
            env_regions.append((pos_env[0], pos_env[1], pos_env[2], r_env, rgba))
            
        self._regions = env_regions
        self._region_num = len(env_regions)

        # Check if we need to rebuild (more regions than currently allocated)
        current_capacity = (
            len(self._region_geoms_list) if hasattr(self, "_region_geoms_list") else 0
        )

        if self._region_num != current_capacity:
            # Number of regions changed, force rebuild on next reset
            self._mjcf_model = None
            self.mark_dirty()
            return

        # Update the MuJoCo model if it has been initialized
        if hasattr(self, "_model") and self._model is not None:
            for i, region in enumerate(env_regions):
                x, y, z, r, rgba = region

                # Update position
                body_id = self._model.body(f"region_{i}").id
                self._model.body_pos[body_id] = (x, y, z)

                # Update size (radius)
                geom_id = self._model.geom(f"region_{i}").id
                self._model.geom_size[geom_id] = (r, 0, 0)

                # Update color
                self._model.geom_rgba[geom_id] = rgba

    def _sample_regions(self, region_num):
        if self._regions is not None:
            return self._regions

        regions = []

        # Only support sphere regions for now.
        if self._region_type == "sphere":
            safe_margin = self._sample_regions_params["safe_margin"]
            workspace_bound = self._workspace_bounds
            x_range = (
                workspace_bound[0][0] + safe_margin,
                workspace_bound[1][0] - safe_margin,
            )
            y_range = (
                workspace_bound[0][1] + safe_margin,
                workspace_bound[1][1] - safe_margin,
            )
            z_range = (
                workspace_bound[0][2] + safe_margin,
                workspace_bound[1][2] - safe_margin,
            )
            workspace_size_avg = np.mean(workspace_bound[1] - workspace_bound[0])
            r_range = (
                workspace_size_avg * self._sample_regions_params["r_range"][0],
                workspace_size_avg * self._sample_regions_params["r_range"][1],
            )
            cmaps = self._generate_colors(region_num)
            rng_state = np.random.get_state()
            np.random.seed(self._region_sample_seed)
            region_generated = 0
            stuck_counter = 0
            while region_generated < region_num:
                x = np.random.uniform(x_range[0], x_range[1])
                y = np.random.uniform(y_range[0], y_range[1])
                z = np.random.uniform(z_range[0], z_range[1])
                r = np.random.uniform(r_range[0], r_range[1])
                r = min(r, x - workspace_bound[0][0])
                r = min(r, workspace_bound[1][0] - x)
                r = min(r, y - workspace_bound[0][1])
                r = min(r, workspace_bound[1][1] - y)
                r = min(r, z - workspace_bound[0][2])
                r = min(r, workspace_bound[1][2] - z)
                if r <= 0.02:
                    continue
                if self._nonoverlapping_regions:
                    # Check for overlaps with existing regions.
                    overlap = False
                    for region in regions:
                        dist = np.linalg.norm(
                            np.array([x, y, z]) - np.array(region[:3])
                        )
                        if dist < r + region[3]:
                            overlap = True
                            break
                    if overlap:
                        stuck_counter += 1
                        if stuck_counter > 1000:
                            print(
                                "Warning: Stuck in region sampling, resetting regions."
                            )
                            regions.clear()
                            region_generated = 0
                            stuck_counter = 0
                        continue
                rgba = (
                    *cmaps[region_generated],
                    self._sample_regions_params["alpha"],
                )
                regions.append((x, y, z, r, rgba))
                region_generated += 1
            np.random.set_state(rng_state)
        else:
            raise NotImplementedError(
                f"Region type {self._region_type} not implemented."
            )
        self._regions = regions
        return regions

    def get_regions_info(self):
        """Get the list of regions in the environment in observation space.

        Returns:
            List of regions, each region is (x, y, z, r, rgba)
        """
        if hasattr(self, "_regions_obs") and self._regions_obs is not None:
            return self._regions_obs
        
        if self._regions is None:
            return []

        # Convert from Env Space to Obs Space
        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0
        
        obs_regions = []
        for region in self._regions:
            x, y, z, r, rgba = region
            pos_env = np.array([x, y, z])
            pos_obs = (pos_env - xyz_center) * xyz_scaler
            r_obs = r * xyz_scaler
            obs_regions.append((pos_obs[0], pos_obs[1], pos_obs[2], r_obs, rgba))
            
        return obs_regions

    def set_tasks(self):
        pass

    def add_objects(self, arena_mjcf):
        # Add tl scene.
        regions = self._sample_regions(self._region_num)

        # Add `num_regions` regions to the scene.
        for i, region in enumerate(regions):
            region_sphere_mjcf = mjcf.from_path(
                (self._desc_dir / "region_sphere.xml").as_posix()
            )
            x, y, z, r, rgba = region
            region_sphere_mjcf.find("body", "region_template").pos = (x, y, z)
            region_sphere_mjcf.find("geom", "region_geom").size = (r, 0, 0)
            # Ensure rgba is a string of space-separated values for MJCF
            if isinstance(rgba, (list, tuple, np.ndarray)):
                rgba_str = " ".join(map(str, rgba))
            else:
                rgba_str = str(rgba)
            region_sphere_mjcf.find("geom", "region_geom").rgba = rgba_str
            region_sphere_mjcf.find("body", "region_template").name = f"region_{i}"
            region_sphere_mjcf.find("geom", "region_geom").name = f"region_{i}"
            arena_mjcf.include_copy(region_sphere_mjcf)

        # Save region geoms.
        self._region_geoms_list = []
        for i in range(self._region_num):
            self._region_geoms_list.append(
                arena_mjcf.find("body", f"region_{i}").find_all("geom")
            )

        # Add cameras.
        cameras = {
            "front": {
                "pos": (1.287, 0.000, 0.509),
                "xyaxes": (0.000, 1.000, 0.000, -0.342, 0.000, 0.940),
            },
            "front_pixels": {
                "pos": (1.053, -0.014, 0.639),
                "xyaxes": (0.000, 1.000, 0.000, -0.628, 0.001, 0.778),
            },
            "top": {
                "pos": (0.400, 0.000, 1.000),
                "xyaxes": (1.000, 0.000, 0.000, 0.000, 1.000, 0.000),
            },
            "left": {
                "pos": (0.400, -0.700, 0.509),
                "xyaxes": (1.000, 0.000, 0.000, 0.000, 0.342, 0.940),
            },
        }
        for camera_name, camera_kwargs in cameras.items():
            arena_mjcf.worldbody.add("camera", name=camera_name, **camera_kwargs)

    def post_compilation_objects(self):
        pass
        # # Cube geom IDs.
        # self._cube_geom_ids_list = [
        #     [self._model.geom(geom.full_identifier).id for geom in cube_geoms]
        #     for cube_geoms in self._cube_geoms_list
        # ]
        # self._cube_target_mocap_ids = [
        #     self._model.body(f"object_target_{i}").mocapid[0]
        #     for i in range(self._num_cubes)
        # ]
        # self._cube_target_geom_ids_list = [
        #     [self._model.geom(geom.full_identifier).id for geom in cube_target_geoms]
        #     for cube_target_geoms in self._cube_target_geoms_list
        # ]

    def initialize_episode(self):
        self._data.qpos[self._arm_joint_ids] = self._home_qpos
        mujoco.mj_kinematics(self._model, self._data)

        if self._mode == "data_collection":
            raise NotImplementedError("Data collection mode not implemented for TLEnv.")
            # # Randomize the scene.
            # self.initialize_arm()

            # # Randomize object positions and orientations.
            # for i in range(self._num_cubes):
            #     xy = self.np_random.uniform(*self._object_sampling_bounds)
            #     obj_pos = (*xy, 0.02)
            #     yaw = self.np_random.uniform(0, 2 * np.pi)
            #     obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
            #     self._data.joint(f"object_joint_{i}").qpos[:3] = obj_pos
            #     self._data.joint(f"object_joint_{i}").qpos[3:] = obj_ori

            # # Set a new target.
            # self.set_new_target(return_info=False)
        else:
            # # Set object positions and orientations based on the current task.
            # init_xyzs = self.cur_task_info["init_xyzs"].copy()[permutation]
            # goal_xyzs = self.cur_task_info["goal_xyzs"].copy()[permutation]

            # # First, force set the current scene to the goal state to obtain the goal observation.
            saved_qpos = self._data.qpos.copy()
            saved_qvel = self._data.qvel.copy()
            # self.initialize_arm()
            # for i in range(self._num_cubes):
            #     self._data.joint(f"object_joint_{i}").qpos[:3] = goal_xyzs[i]
            #     self._data.joint(f"object_joint_{i}").qpos[
            #         3:
            #     ] = lie.SO3.identity().wxyz.tolist()
            #     self._data.mocap_pos[self._cube_target_mocap_ids[i]] = goal_xyzs[i]
            #     self._data.mocap_quat[self._cube_target_mocap_ids[i]] = (
            #         lie.SO3.identity().wxyz.tolist()
            #     )
            # mujoco.mj_forward(self._model, self._data)

            # # Do a few random steps to make the scene stable.
            # for _ in range(2):
            #     self.step(self.action_space.sample())

            # # Save the goal observation.
            # self._cur_goal_ob = (
            #     self.compute_oracle_observation()
            #     if self._use_oracle_rep
            #     else self.compute_observation()
            # )
            # if self._render_goal:
            #     self._cur_goal_rendered = self.render()
            # else:
            #     self._cur_goal_rendered = None

            # Now, do the actual reset.
            self._data.qpos[:] = saved_qpos
            self._data.qvel[:] = saved_qvel
            self.initialize_arm()
            # for i in range(self._num_cubes):
            #     # Randomize the position and orientation of the cube slightly.
            #     obj_pos = init_xyzs[i].copy()
            #     obj_pos[:2] += self.np_random.uniform(-0.01, 0.01, size=2)
            #     self._data.joint(f"object_joint_{i}").qpos[:3] = obj_pos
            #     yaw = self.np_random.uniform(0, 2 * np.pi)
            #     obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
            #     self._data.joint(f"object_joint_{i}").qpos[3:] = obj_ori
            #     self._data.mocap_pos[self._cube_target_mocap_ids[i]] = goal_xyzs[i]
            #     self._data.mocap_quat[self._cube_target_mocap_ids[i]] = (
            #         lie.SO3.identity().wxyz.tolist()
            #     )

        # Forward kinematics to update site positions.
        self.pre_step()
        mujoco.mj_forward(self._model, self._data)
        self.post_step()

        self._success = False

    def set_new_target(self, return_info=True, p_stack=0.5):
        """Set a new random target for data collection.

        Args:
            return_info: Whether to return the observation and reset info.
            p_stack: Probability of stacking the target block on top of another block when there are multiple blocks.
        """
        raise NotImplementedError("Data collection mode not implemented for TLEnv.")

        assert self._mode == "data_collection"

        block_xyzs = np.array(
            [
                self._data.joint(f"object_joint_{i}").qpos[:3]
                for i in range(self._num_cubes)
            ]
        )

        # Compute the top blocks.
        top_blocks = []
        for i in range(self._num_cubes):
            for j in range(self._num_cubes):
                if i == j:
                    continue
                if (
                    block_xyzs[j][2] > block_xyzs[i][2]
                    and np.linalg.norm(block_xyzs[i][:2] - block_xyzs[j][:2]) < 0.02
                ):
                    break
            else:
                top_blocks.append(i)

        # Pick one of the top cubes as the target.
        self._target_block = self.np_random.choice(top_blocks)

        stack = len(top_blocks) >= 2 and self.np_random.uniform() < p_stack
        if stack:
            # Stack the target block on top of another block.
            block_idx = self.np_random.choice(
                list(set(top_blocks) - {self._target_block})
            )
            block_pos = self._data.joint(f"object_joint_{block_idx}").qpos[:3]
            tar_pos = np.array([block_pos[0], block_pos[1], block_pos[2] + 0.04])
        else:
            # Randomize target position.
            xy = self.np_random.uniform(*self._target_sampling_bounds)
            tar_pos = (*xy, 0.02)
        # Randomize target orientation.
        yaw = self.np_random.uniform(0, 2 * np.pi)
        tar_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()

        # Only show the target block.
        for i in range(self._num_cubes):
            if i == self._target_block:
                # Set the target position and orientation.
                self._data.mocap_pos[self._cube_target_mocap_ids[i]] = tar_pos
                self._data.mocap_quat[self._cube_target_mocap_ids[i]] = tar_ori
            else:
                # Move the non-target blocks out of the way.
                self._data.mocap_pos[self._cube_target_mocap_ids[i]] = (0, 0, -0.3)
                self._data.mocap_quat[self._cube_target_mocap_ids[i]] = (
                    lie.SO3.identity().wxyz.tolist()
                )

        # Set the target colors.
        for i in range(self._num_cubes):
            if self._visualize_info and i == self._target_block:
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.2
            else:
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.0

        if return_info:
            return self.compute_observation(), self.get_reset_info()

    def _compute_successes(self):
        """Compute object successes."""
        raise NotImplementedError("Success computation not implemented for TLEnv.")
        cube_successes = []
        for i in range(self._num_cubes):
            obj_pos = self._data.joint(f"object_joint_{i}").qpos[:3]
            tar_pos = self._data.mocap_pos[self._cube_target_mocap_ids[i]]
            if np.linalg.norm(obj_pos - tar_pos) <= 0.04:
                cube_successes.append(True)
            else:
                cube_successes.append(False)

        return cube_successes

    def post_step(self):
        return
        # Check if the cubes are in the target positions.
        cube_successes = self._compute_successes()
        if self._mode == "data_collection":
            self._success = cube_successes[self._target_block]
        else:
            self._success = all(cube_successes)

        # Adjust the colors of the cubes based on success.
        for i in range(self._num_cubes):
            if self._visualize_info and (
                self._mode == "task" or i == self._target_block
            ):
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.2
            else:
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.0

            if self._visualize_info and cube_successes[i]:
                for gid in self._cube_geom_ids_list[i]:
                    self._model.geom(gid).rgba[:3] = self._cube_success_colors[i, :3]
            else:
                for gid in self._cube_geom_ids_list[i]:
                    self._model.geom(gid).rgba[:3] = self._cube_colors[i, :3]

    def add_object_info(self, ob_info):
        return
        # Cube positions and orientations.
        for i in range(self._num_cubes):
            ob_info[f"privileged/block_{i}_pos"] = (
                self._data.joint(f"object_joint_{i}").qpos[:3].copy()
            )
            ob_info[f"privileged/block_{i}_quat"] = (
                self._data.joint(f"object_joint_{i}").qpos[3:].copy()
            )
            ob_info[f"privileged/block_{i}_yaw"] = np.array(
                [
                    lie.SO3(
                        wxyz=self._data.joint(f"object_joint_{i}").qpos[3:]
                    ).compute_yaw_radians()
                ]
            )

        if self._mode == "data_collection":
            # Target cube info.
            ob_info["privileged/target_task"] = self._target_task

            target_mocap_id = self._cube_target_mocap_ids[self._target_block]
            ob_info["privileged/target_block"] = self._target_block
            ob_info["privileged/target_block_pos"] = self._data.mocap_pos[
                target_mocap_id
            ].copy()
            ob_info["privileged/target_block_yaw"] = np.array(
                [
                    lie.SO3(
                        wxyz=self._data.mocap_quat[target_mocap_id]
                    ).compute_yaw_radians()
                ]
            )

    def compute_observation(self):
        if self._ob_type == "pixels":
            return self.get_pixel_observation()
        else:
            xyz_center = np.array([0.425, 0.0, 0.0])
            xyz_scaler = 10.0
            gripper_scaler = 3.0

            ob_info = self.compute_ob_info()
            ob = [
                ob_info["proprio/joint_pos"],
                ob_info["proprio/joint_vel"],
                (ob_info["proprio/effector_pos"] - xyz_center) * xyz_scaler,
                np.cos(ob_info["proprio/effector_yaw"]),
                np.sin(ob_info["proprio/effector_yaw"]),
                ob_info["proprio/gripper_opening"] * gripper_scaler,
                ob_info["proprio/gripper_contact"],
            ]

            return np.concatenate(ob)

    def compute_oracle_observation(self):
        """Return the oracle goal representation of the current state."""
        return
        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0

        ob_info = self.compute_ob_info()
        ob = []
        for i in range(self._num_cubes):
            ob.append((ob_info[f"privileged/block_{i}_pos"] - xyz_center) * xyz_scaler)

        return np.concatenate(ob)

    def compute_reward(self, ob, action):
        return
        if self._reward_task_id is None:
            return super().compute_reward(ob, action)

        # Compute the reward based on the task.
        successes = self._compute_successes()
        reward = float(sum(successes) - len(successes))
        return reward
