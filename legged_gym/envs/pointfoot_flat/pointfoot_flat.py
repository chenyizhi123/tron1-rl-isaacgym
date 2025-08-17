import torch
import numpy as np
import os
import math

from isaacgym.torch_utils import *
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
)
from .pointfoot_flat_config import BipedCfgPF

class BipedPF(BaseTask):
    
    def __init__(
        self, cfg: BipedCfgPF, sim_params, physics_engine, sim_device, headless
    ):
        """Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None

        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2

        self.group_idx = torch.arange(0, self.cfg.env.num_envs)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)

        Returns:
            obs (torch.Tensor): Tensor of shape (num_envs, num_observations_per_env)
            rewards (torch.Tensor): Tensor of shape (num_envs)
            dones (torch.Tensor): Tensor of shape (num_envs)
        """
        self._action_clip(actions)
        # step physics and render each frame
        self.render()
        self.pre_physics_step()
        for _ in range(self.cfg.control.decimation):
            self.action_fifo = torch.cat(
                (self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1
            )
            self.envs_steps_buf += 1
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs), self.action_delay_idx, :]
            ).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,
            self.commands[:, :3] * self.commands_scale,
            self.critic_obs_buf # make sure critic_obs update in every for loop
        )

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = (
            self.command_ranges["lin_vel_x"][env_ids, 1]
            - self.command_ranges["lin_vel_x"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_x"
        ][
            env_ids, 0
        ]
        self.commands[env_ids, 1] = (
            self.command_ranges["lin_vel_y"][env_ids, 1]
            - self.command_ranges["lin_vel_y"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_y"
        ][
            env_ids, 0
        ]
        self.commands[env_ids, 2] = (
            self.command_ranges["ang_vel_yaw"][env_ids, 1]
            - self.command_ranges["ang_vel_yaw"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "ang_vel_yaw"
        ][
            env_ids, 0
        ]
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        # set small commands to zero
        # self.commands[env_ids, :2] *= (
        #     torch.norm(self.commands[env_ids, :2], dim=1) > self.cfg.commands.min_norm
        # ).unsqueeze(1)
        zero_command_idx = (
            (
                torch_rand_float(0, 1, (len(env_ids), 1), device=self.device)
                > self.cfg.commands.zero_command_prob
            )
            .squeeze(1)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.commands[zero_command_idx, :3] = 0
        if self.cfg.commands.heading_command:
            forward = quat_apply(
                self.base_quat[zero_command_idx], self.forward_vec[zero_command_idx]
            )
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[zero_command_idx, 3] = heading

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale

        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = (
                self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
                - self.d_gains * self.dof_vel
            )
        elif control_type == "V":
            torques = (
                self.p_gains * (actions_scaled - self.dof_vel)
                - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            )
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(
            torques * self.torques_scale, -self.torque_limits, self.torque_limits
        )

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = (
            noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        )
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:12] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[12:18] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[18:] = 0.0  # previous actions
        return noise_vec
    
    def reset_idx(self, env_ids):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum:
            time_out_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
            self.update_command_curriculum(time_out_env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self._resample_gaits(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history[env_ids] = 0
        obs_buf, _ = self.compute_group_observations()
        self.obs_history[env_ids] = obs_buf[env_ids].repeat(1, self.obs_history_length)
        self.gait_indices[env_ids] = 0
        self.fail_buf[env_ids] = 0
        self.action_fifo[env_ids] = 0
        self.dof_pos_int[env_ids] = 0
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["group_terrain_level"] = torch.mean(
                self.terrain_levels[self.group_idx].float()
            )
            self.extras["episode"]["group_terrain_level_stair_up"] = torch.mean(
                self.terrain_levels[self.stair_up_idx].float()
            )
        if self.cfg.terrain.curriculum and self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.smooth_slope_idx, 1].float()
            )
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf | self.edge_reset_buf

    def compute_group_observations(self):
        # note that observation noise need to modified accordingly !!!
        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                self.clock_inputs_sin.view(self.num_envs, 1),
                self.clock_inputs_cos.view(self.num_envs, 1),
                self.gaits,
            ),
            dim=-1,
        )
        
        # 为critic添加跳跃相关的额外观测信息
        # 1. 足部接触状态 (2维)
        foot_contacts = (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0).float()
        
        # 2. 足部高度信息 (2维) - 使用足部Z坐标相对于地面的高度
        foot_z_heights = self.foot_positions[:, :, 2] - self._get_foot_heights()
        foot_heights_normalized = torch.clip(foot_z_heights, 0, 1)   # 归一化足部高度
        
        # 3. 基座垂直加速度 (1维)
        base_z_acc = ((self.base_lin_vel[:, 2] - getattr(self, 'last_base_z_vel', torch.zeros_like(self.base_lin_vel[:, 2]))) / self.dt).unsqueeze(1)
        self.last_base_z_vel = self.base_lin_vel[:, 2].clone()
        
        # 4. 接触力大小 (2维)
        contact_force_magnitudes = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)   # 缩放
        
        # 5. 步态相位信息 (2维) - 增强版
        gait_phase_enhanced = self.desired_contact_states  # 期望的接触状态
        
        # 6. 腾空时间信息 (1维)
        both_feet_airborne = (torch.sum(foot_contacts, dim=1) == 0).float().unsqueeze(1)
        
        # 7. 基座高度相对于目标的偏差 (1维)
        base_height_error = (self.base_position[:, 2] - self.cfg.rewards.base_height_target).unsqueeze(1)
        
        # 8. 躯干倾斜度 (2维)
        trunk_tilt = self.projected_gravity[:, :2]  # 重复使用已有的重力投影
        
        critic_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,  # 基础线性速度 (3维)
            obs_buf,  # 原始观测 (30维)
            foot_contacts,  # 足部接触状态 (2维)
            foot_heights_normalized,  # 足部高度 (2维)
            base_z_acc,  # 垂直加速度 (1维)
            contact_force_magnitudes,  # 接触力大小 (2维)
            gait_phase_enhanced,  # 步态相位 (2维)
            both_feet_airborne,  # 腾空状态 (1维)
            base_height_error,  # 高度偏差 (1维)
            trunk_tilt,  # 躯干倾斜 (2维)
        ), dim=-1)
        
        return obs_buf, critic_obs_buf
    
    # --------------------------- reward functions---------------------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        reward = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return reward

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.dof_acc), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.actions - self.last_actions[:, :, 0]), dim=1)

    def _reward_action_smooth(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(
                self.actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1)

    def _reward_keep_balance(self):
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.ang_tracking_sigma)

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states

        reward = 0
        if self.reward_scales["tracking_contacts_shaped_force"] > 0:
            for i in range(len(self.feet_indices)):
                reward += (1 - desired_contact[:, i]) * torch.exp(
                    -foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma)
        else:
            for i in range(len(self.feet_indices)):
                reward += (1 - desired_contact[:, i]) * (
                    1 - torch.exp(-foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma))

        return reward / len(self.feet_indices)

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.foot_velocities, dim=-1)
        desired_contact = self.desired_contact_states
        reward = 0
        if self.reward_scales["tracking_contacts_shaped_vel"] > 0:
            for i in range(len(self.feet_indices)):
                reward += desired_contact[:, i] * torch.exp(
                    -foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma
                )
        else:
            for i in range(len(self.feet_indices)):
                reward += desired_contact[:, i] * (
                    1 - torch.exp(-foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma))
        return reward / len(self.feet_indices)

    def _reward_feet_distance(self):
        # Penalize base height away from target
        feet_distance = torch.norm(self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1)
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1)
        return reward

    def _reward_feet_regulation(self):
        feet_height = self.cfg.rewards.base_height_target * 0.001
        # 计算足部相对于地面的高度
        foot_heights_2d = self.foot_positions[:, :, 2] - self._get_foot_heights()
        foot_heights_2d = torch.clip(foot_heights_2d, 0, 1)
        reward = torch.sum(
            torch.exp(-foot_heights_2d / feet_height)
            * torch.square(torch.norm(self.foot_velocities[:, :, :2], dim=-1)), dim=1)
        return reward

    def _reward_collision(self):
        return torch.sum(
            torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 1.0, dim=1)

    def _reward_foot_landing_vel(self):
        z_vels = self.foot_velocities[:, :, 2]
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        # 计算足部相对于地面的高度
        foot_heights_2d = self.foot_positions[:, :, 2] - self._get_foot_heights()
        foot_heights_2d = torch.clip(foot_heights_2d, 0, 1)
        about_to_land = (foot_heights_2d < self.cfg.rewards.about_landing_threshold) & (~contacts) & (z_vels < 0.0)
        landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
        reward = torch.sum(torch.square(landing_z_vels), dim=1)
        return reward

    def _reward_vertical_impulse(self):
        """鼓励向上的推进力，促进跳跃行为"""
        contact_forces = self.contact_forces[:, self.feet_indices, :]
        vertical_forces = contact_forces[:, :, 2]  # Z方向力
        # 只在接触时计算奖励
        in_contact = torch.norm(contact_forces, dim=-1) > 1.0
        impulse = torch.where(in_contact, vertical_forces, torch.zeros_like(vertical_forces))
        # 鼓励强力向上推进
        reward = torch.sum(torch.clip(impulse - self.cfg.rewards.min_impulse_threshold, 0, None), dim=1)
        return reward

    def _reward_jump_height(self):
        """奖励达到目标跳跃高度"""
        both_feet_airborne = torch.sum(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0, dim=1) == 0
        current_height = self.base_position[:, 2]
        height_above_target = torch.clip(current_height - self.cfg.rewards.target_jump_height, 0, None)
        reward = torch.where(both_feet_airborne, height_above_target, torch.zeros_like(height_above_target))
        return reward

    def _reward_airtime(self):
        """奖励适当的腾空时间"""
        both_feet_airborne = torch.sum(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0, dim=1) == 0
        # 更新腾空时间计数
        self.feet_air_time[:, 0] = torch.where(both_feet_airborne, 
                                               self.feet_air_time[:, 0] + self.dt,
                                               torch.zeros_like(self.feet_air_time[:, 0]))
        # 奖励在目标腾空时间范围内的情况
        target_airtime = self.cfg.rewards.target_airtime
        airtime_error = torch.abs(self.feet_air_time[:, 0] - target_airtime)
        reward = torch.exp(-airtime_error / self.cfg.rewards.airtime_sigma) * both_feet_airborne.float()
        return reward

    def _reward_landing_stability(self):
        """奖励平稳着陆"""
        # 检测着陆瞬间
        current_contacts = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0
        was_airborne = torch.sum(self.last_contacts, dim=1) == 0
        just_landed = torch.sum(current_contacts, dim=1) > 0
        landing_moment = was_airborne & just_landed
        
        # 着陆时的基座稳定性
        base_angular_vel = torch.norm(self.base_ang_vel, dim=1)
        base_lin_vel_z = torch.abs(self.base_lin_vel[:, 2])
        stability_penalty = base_angular_vel + base_lin_vel_z
        
        reward = torch.where(landing_moment, 
                           torch.exp(-stability_penalty / self.cfg.rewards.landing_stability_sigma),
                           torch.zeros_like(stability_penalty))
        return reward

    def _reward_jump_frequency(self):
        """鼓励合适的跳跃频率"""
        both_feet_airborne = torch.sum(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0, dim=1) == 0
        # 简单的跳跃频率奖励，基于步态频率
        target_freq = self.gaits[:, 0]  # 使用步态频率作为目标跳跃频率
        freq_reward = torch.where(both_feet_airborne, 
                                target_freq / self.cfg.rewards.max_jump_frequency,
                                torch.zeros_like(target_freq))
        return freq_reward

    def _reward_forward_jump_progress(self):
        """鼓励向前跳跃的进展"""
        both_feet_airborne = torch.sum(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0, dim=1) == 0
        forward_vel = self.base_lin_vel[:, 0]
        # 在腾空时奖励向前速度
        reward = torch.where(both_feet_airborne & (forward_vel > 0), 
                           forward_vel * self.cfg.rewards.forward_jump_scale,
                           torch.zeros_like(forward_vel))
        return reward

    def check_termination(self):
        """重写终止条件以适应跳跃行为"""
        # 基础失败条件 - 适应跳跃行为
        
        # 1. 不良身体部位接触 - 提高接触力阈值，因为跳跃时力更大
        fail_buf = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            ) > 15.0,  # 从10.0提高到15.0
            dim=1,
        )
        
        # 2. 严重倾斜 - 放宽一些，因为跳跃时可能有姿态变化
        fail_buf |= self.projected_gravity[:, 2] > 0.2  # 从-0.1改为0.2，允许更大倾斜
        
        # 3. 跳跃特定失败条件
        # 过度向后倾斜（危险）
        fail_buf |= self.projected_gravity[:, 0] > 0.8  # 过度后倾
        
        # 基座位置过低（摔倒）
        fail_buf |= self.base_position[:, 2] < 0.3  # 基座高度过低
        
        # 过度侧倾（危险）
        fail_buf |= torch.abs(self.projected_gravity[:, 1]) > 0.8  # 过度侧倾
        
        # 4. 跳跃高度安全检查 - 防止过度跳跃
        fail_buf |= self.base_position[:, 2] > 1.8  # 防止跳得过高而失控
        
        # 5. 长时间腾空检查（可能卡住）
        both_feet_airborne = torch.sum(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0, dim=1) == 0
        long_airtime = self.feet_air_time[:, 0] > 3.0  # 腾空超过0.8秒
        fail_buf |= both_feet_airborne & long_airtime
        
        self.fail_buf += fail_buf
        
        # 超时条件
        self.time_out_buf = (
            self.episode_length_buf > self.max_episode_length
        )
        
        # 功率限制
        self.power_limit_out_buf = (
            torch.sum(self.power, dim=1) > self.cfg.control.max_power
        )
        
        # 地形边界检查
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.edge_reset_buf = self.base_position[:, 0] > self.terrain_x_max - 1
            self.edge_reset_buf |= self.base_position[:, 0] < self.terrain_x_min + 1
            self.edge_reset_buf |= self.base_position[:, 1] > self.terrain_y_max - 1
            self.edge_reset_buf |= self.base_position[:, 1] < self.terrain_y_min + 1
        
        # 最终重置决策
        self.reset_buf = (
            (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt)
            | self.time_out_buf
            | self.edge_reset_buf
            # | self.power_limit_out_buf  # 暂时禁用功率限制，跳跃需要更大功率
        )