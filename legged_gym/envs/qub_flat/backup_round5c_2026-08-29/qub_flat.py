import torch
import numpy as np
import os
import math

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
from .qub_flat_config import QUBCfg

class QUBFlat(BaseTask):
    
    def __init__(
        self, cfg: QUBCfg, sim_params, physics_engine, sim_device, headless
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
            print("=" * 50)
            print("Torque Limits:")
            print(getattr(self, 'torque_limits', 'N/A'))
            print("\nP Gains:")
            print(getattr(self, 'p_gains', 'N/A'))
            print("D Gains:", getattr(self, 'd_gains', 'N/A'))
            print("=" * 50)

        self._init_buffers()

        # feet_air_time을 방향 무관하게 "뗐다 착지"만 보면 옆으로 다리를 벌리는 것도
        # farming 가능해서(2026-08-28 실측으로 실제 확인됨), 스윙 시작 시점의 발 위치를
        # 기록해뒀다가 착지 시 커맨드 방향으로의 변위를 계산하는 데 씀.
        self.feet_liftoff_pos = self.foot_positions[:, :, :2].clone()

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
        """Randomly select commands of some environments

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
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        if self.cfg.commands.curriculum:
            time_out_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
            self.update_command_curriculum(time_out_env_ids)

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self._resample_gaits(env_ids)

        self.last_actions[env_ids] = 0.0
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.feet_liftoff_pos[env_ids] = self.foot_positions[env_ids, :, :2]
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

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
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
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf | self.edge_reset_buf

    def _update_reward_scales(self):
        """학습 진행도에 따라 보상 가중치 동적 조정"""
        progress = self.episode_length_buf / self.max_episode_length
    
        if progress.mean() < 0.3:
            scale_factor = 1.0
        elif progress.mean() < 0.7:
            scale_factor = 1.2
            self.reward_scales['orientation'] *= 1.1
        else:
            scale_factor = 1.5
            self.reward_scales['action_smooth'] *= 1.1
    
        return scale_factor

    def compute_group_observations(self):
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
        critic_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel, self.obs_buf), dim=-1)
        
        return obs_buf, critic_obs_buf

    def _post_physics_step_callback(self):
        # feet_air_time/last_contacts 버퍼는 base_task.py에 선언(및 reset_idx에서 초기화)만
        # 되어있고 실제로 갱신하는 코드가 어디에도 없었음(죽은 인프라). 여기서 살려서
        # "발을 떼서 착지"할 때마다 보상하는 _reward_feet_air_time에 쓸 값을 계산해둠.
        # tracking_lin_vel/contacts_shaped 만으로는 "실제로 발을 떼는 행동" 자체에 대한
        # 직접적인 유인이 약해서(추상적인 결과 보상), 한 발짝 떼고 다시 정지하는 문제를
        # 직접적으로 풀어보려는 목적. base_task.py는 다른 로봇(pointfoot 등)도 같이 쓰므로
        # 여기 qub_flat.py에서만 override.
        super()._post_physics_step_callback()

        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)

        # 스윙이 막 시작되는(직전엔 접촉, 지금은 비접촉) 순간의 발 위치를 저장.
        # 착지 시 이 위치 대비 변위를 계산해서 "실제로 커맨드 방향으로 나아갔는지"를
        # 판정하는 데 씀 — 이게 없으면 옆으로 트는 스텝도 air_time 보상을 받아감.
        liftoff = self.last_contacts & (~contact)
        self.feet_liftoff_pos[liftoff] = self.foot_positions[:, :, :2][liftoff]

        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt

        # 착지 시점 발 위치 - 스윙 시작 시점 발 위치 = 이번 스텝의 실제 변위(월드 프레임).
        # base_lin_vel과 동일하게 base_quat으로 바디 프레임으로 회전시켜 commands(바디
        # 프레임, forward=x)와 같은 기준으로 비교. quat_rotate_inverse는 배치 차원이
        # q/v 사이에 정확히 일치해야 해서 (num_envs, num_feet)를 하나로 펴서 계산.
        num_feet = self.feet_indices.shape[0]
        step_disp_world = self.foot_positions[:, :, :2] - self.feet_liftoff_pos
        step_disp_world_3d = torch.cat(
            (step_disp_world, torch.zeros_like(step_disp_world[:, :, :1])), dim=-1
        )
        quat_per_foot = self.base_quat.repeat_interleave(num_feet, dim=0)
        step_disp_body = quat_rotate_inverse(
            quat_per_foot, step_disp_world_3d.reshape(-1, 3)
        ).reshape(self.num_envs, num_feet, 3)[:, :, :2]

        cmd_dir = self.commands[:, :2]
        cmd_dir_unit = cmd_dir / torch.norm(cmd_dir, dim=1, keepdim=True).clamp(min=1e-6)
        forward_progress = torch.sum(step_disp_body * cmd_dir_unit.unsqueeze(1), dim=-1)  # (N, feet)

        # 최소 3cm는 커맨드 방향으로 나아가야 "걸음"으로 인정 (옆으로 트는 것 배제)
        min_forward_step = 0.03
        direction_ok = (forward_progress > min_forward_step) & first_contact

        # 목표 스윙 시간(약 0.3초, 이 게이트 설정 durations=[0.5,0.7]/frequencies=[0.8,1.2]
        # 기준 예상 스윙 시간과 비슷한 수준) 이상 발을 띄웠다가 "전진 방향으로" 착지하면
        # 양의 보상, 방향이 안 맞으면(옆으로만 이동) 이번 착지는 보상에서 제외.
        target_swing_time = 0.3
        air_time_reward = torch.sum(
            (self.feet_air_time - target_swing_time) * direction_ok, dim=1
        )
        # 명령이 사실상 0인 로봇(제자리 유지가 목표)에는 발 구르기를 강요하지 않음
        air_time_reward *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self._feet_air_time_reward = air_time_reward

        self.feet_air_time *= ~contact_filt

    def post_physics_step(self):
        # ⚠️ 부모 클래스(BaseTask)의 post_physics_step을 반드시 먼저 호출해야 함.
        # 여기에 리셋 판정(check_termination), reset_idx 호출, gait clock 갱신,
        # 보상 계산 등 핵심 로직이 들어있음. 이걸 빼먹으면 리스폰도 안 되고
        # 발 구르기(gait)도 갱신되지 않음.
        super().post_physics_step()

        # 제자리 걷기 메트릭 기록 및 엑스트라 추가 (부가 기능이므로 그 다음에 실행)
        if hasattr(self, 'extras'):
            walking_metrics = self._get_walking_metrics()
            self.extras['walking_metrics'] = walking_metrics

        # TensorBoard 로깅용 누적 메트릭 추가
        if hasattr(self, 'episode_sums'):
            metrics = self._get_walking_metrics()
            if 'height_stability' in self.episode_sums:
                self.episode_sums['height_stability'] += metrics['height_std']
            if 'lin_vel_x' in self.episode_sums:
                self.episode_sums['lin_vel_x'] += metrics['lin_vel_x_mean']
            if 'lin_vel_y' in self.episode_sums:
                self.episode_sums['lin_vel_y'] += metrics['lin_vel_y_mean']
            if 'contact_alternation' in self.episode_sums:
                self.episode_sums['contact_alternation'] += metrics['contact_alternation']

        # 비정상 상태 감지 검증기 호출
        if not self.headless and len(self.episode_length_buf) > 0 and self.episode_length_buf[0].item() % 100 == 0:
            self._validate_walking_stability()

    def _get_walking_metrics(self):
        """제자리 걷기의 품질을 평가하는 메트릭 계산"""
        metrics = {
            'height_std': torch.std(self.root_states[:, 2]),
            'height_mean': torch.mean(self.root_states[:, 2]),
            'lin_vel_x_mean': torch.mean(torch.abs(self.base_lin_vel[:, 0])),
            'lin_vel_y_mean': torch.mean(torch.abs(self.base_lin_vel[:, 1])),
            'lin_vel_z_mean': torch.mean(torch.abs(self.base_lin_vel[:, 2])),
            'roll_std': torch.std(self.projected_gravity[:, 0]),
            'pitch_std': torch.std(self.projected_gravity[:, 1]),
            'contact_alternation': self._compute_contact_alternation(),
            'feet_height_diff': torch.mean(
                torch.abs(self.foot_heights[:, 0] - self.foot_heights[:, 1])
            ),
        }
        return metrics

    def _compute_contact_alternation(self):
        """양발이 정확히 교대로 접촉하는 정도 계산 (0~1)"""
        contact = self.contact_forces[:, self.feet_indices, 2] > 0.1
        alternation_score = torch.ones_like(contact[:, 0], dtype=torch.float)
    
        for i in range(self.num_envs):
            ideal = (contact[i, 0] != contact[i, 1]).float()
            alternation_score[i] = ideal
    
        return torch.mean(alternation_score)

    def _validate_walking_stability(self):
        """제자리 걷기 중 비정상 상태 감지"""
        if hasattr(self, 'contact_forces') and hasattr(self, 'knee_indices'):
            knee_contacts = self.contact_forces[:, self.knee_indices, 2] > 0.1
            if knee_contacts.any():
                print(f"⚠️ WARNING: {knee_contacts.sum()} robots touching ground with knee")
    
        if torch.abs(self.base_lin_vel[:, 2]).max() > 2.0:
            print(f"⚠️ WARNING: Abnormal vertical velocity: {self.base_lin_vel[:, 2].max()}")
    
        if hasattr(self, 'last_base_position'):
            height_change = torch.abs(self.root_states[:, 2] - self.last_base_position[:, 2])
            if height_change.max() > 0.05:
                print(f"⚠️ WARNING: Large height change: {height_change.max()}")
    
        if torch.norm(self.projected_gravity[:, :2], dim=1).max() > 0.2:
            print(f"⚠️ WARNING: Robot tilted: {torch.norm(self.projected_gravity[:, :2], dim=1).max()}")
    
    # --------------------------- reward functions---------------------------
    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        reward = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return reward

    def _reward_base_height(self):
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square(self.dof_acc), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.actions - self.last_actions[:, :, 0]), dim=1)

    def _reward_action_smooth(self):
        return torch.sum(
            torch.square(
                self.actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1)

    def _reward_keep_balance(self):
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_pos(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
    
    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
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
        # 원래 min_feet_distance(다리 겹침 방지)만 있고 상한이 없어서, 다리를 옆으로
        # 최대한 벌리는 게 페널티 없이 안정성만 버는 도피처였음(2026-08-28 실측/시각
        # 확인). 같은 저장소의 wheelfoot_flat.py에 이미 있는 상한 패턴을 그대로 적용.
        feet_distance = torch.norm(self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1)
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1) + \
                 torch.clip(feet_distance - self.cfg.rewards.max_feet_distance, 0, 1)
        return reward

    def _reward_feet_regulation(self):
        feet_height = self.cfg.rewards.base_height_target * 0.001
        reward = torch.sum(
            torch.exp(-self.foot_heights / feet_height)
            * torch.square(torch.norm(self.foot_velocities[:, :, :2], dim=-1)), dim=1)
        return reward

    def _reward_collision(self):
        return torch.sum(
            torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 1.0, dim=1)

    def _reward_foot_landing_vel(self):
        z_vels = self.foot_velocities[:, :, 2]
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        about_to_land = (self.foot_heights < self.cfg.rewards.about_landing_threshold) & (~contacts) & (z_vels < 0.0)
        landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
        reward = torch.sum(torch.square(landing_z_vels), dim=1)
        return reward

    def _reward_feet_swing_height(self):
        """스윙 구간(desired_contact=0)일 때 발이 목표 높이(swing_height)에 가까워지도록 유도"""
        desired_contact = self.desired_contact_states  # (num_envs, num_feet), 1=stance, 0=swing
        # foot_heights: 발의 지면으로부터 높이
        height_error = torch.square(
            self.foot_heights - self.cfg.rewards.feet_height_target
        )
        # 스윙 구간(desired_contact=0)일 때만 이 오차에 페널티
        reward = torch.sum((1 - desired_contact) * height_error, dim=1)
        return reward

    def _reward_feet_air_time(self):
        """발을 떼서 착지할 때마다 보상 (실제로 발을 드는 행동 자체에 대한 직접적인 유인).
        값은 _post_physics_step_callback에서 미리 계산해둠 (발 접촉 판정을 여기서 다시
        하면 last_contacts 갱신이 두 번 일어나서 상태가 꼬임)."""
        return self._feet_air_time_reward