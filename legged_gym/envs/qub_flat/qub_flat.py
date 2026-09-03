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
from .qub_flat_config import QUBCfg


class QUBFlat(BaseTask):

    def __init__(
        self, cfg: QUBCfg, sim_params, physics_engine, sim_device, headless
    ):
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

        # [P3-c] 좌우 보폭 비대칭 페널티(_reward_step_symmetry)용 버퍼.
        # 발의 '몸 기준 전후 위치'를 보행 주기보다 긴 시정수로 평균 낸 값.
        # alpha=0.01 -> 시정수 100스텝 = 2초 ≈ 3.4 보행주기(1.7Hz 기준).
        self.step_sym_alpha = 0.01
        self.foot_x_ema = torch.zeros(
            self.num_envs, len(self.feet_indices),
            dtype=torch.float, device=self.device, requires_grad=False)

        # ─────────────────────────────────────────────────────────────────
        # [P3-e] 2026-09-02 골반 비틀림 / 착지위치 / 보폭 대칭용 버퍼
        #
        # 왜 필요한가(실측 근거, Sep02_01-49-09_/model_2000, cmd 0.4, 64/64 생존):
        #   URDF가 base_link --torso_yaw_joint(+-45deg)--> pelvis_link --> 양다리 라서
        #   관측/보상의 기준(base_link)과 다리가 달린 몸통(pelvis_link)이 관절 하나를
        #   사이에 두고 분리돼 있다. 그 결과 실제로 이런 일이 벌어지고 있었다:
        #     - torso_yaw DC(시간평균) +3.40도, 개체간 표준편차 0.01도
        #       -> 64대 전부가 똑같이 상체를 비틀어 '고정'. 노이즈가 아니라 학습된 전략.
        #     - 좌우 발 전후위치 차이가 base 기준 5.72cm vs pelvis 기준 7.02cm
        #       -> base 기준으로 재던 기존 step_symmetry는 실제보다 1.30cm(19%) 좋게
        #          보이는 값을 최적화하고 있었다. 골반을 틀면 좌우 오프셋이 전후축으로
        #          섞여 들어가 '보폭을 맞춘 것처럼' 보이게 만들 수 있기 때문.
        #
        # DC와 AC를 나누는 이유: 사람도 걸을 때 골반은 좌우로 돌아간다(AC). 그건 정상이고
        # 각운동량 상쇄에 이롭다. 이상한 것은 한쪽으로 '틀어진 채 고정'된 성분(DC)이다.
        # 시정수 2초(=3.4 보행주기) EMA를 쓰면 1.7Hz 진동은 1/sqrt(1+(w*tau)^2) ~ 0.047배로
        # 눌려 +-1.9도 AC가 +-0.09도만 남는다. 즉 EMA는 DC만 거의 그대로 통과시킨다.
        self.pelvis_body_idx = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], "pelvis_link")
        self.torso_yaw_dof = self.dof_names.index("torso_yaw_joint")
        self.torso_yaw_ema = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # 착지/이지 이벤트용. 이 EMA들은 스텝마다가 아니라 '발이 닿는 순간'에만 갱신되므로
        # (발당 초당 1.7회) alpha가 훨씬 커야 한다. 0.1 -> 시정수 10이벤트 ~ 6초.
        self.event_ema_alpha = 0.1
        n_feet = len(self.feet_indices)
        self.last_foot_contacts = torch.zeros(
            self.num_envs, n_feet, dtype=torch.bool, device=self.device,
            requires_grad=False)
        self.foot_touchdown_x_ema = torch.zeros(
            self.num_envs, n_feet, dtype=torch.float, device=self.device,
            requires_grad=False)
        self.foot_liftoff_x = torch.zeros(
            self.num_envs, n_feet, dtype=torch.float, device=self.device,
            requires_grad=False)
        self.foot_stride_ema = torch.zeros(
            self.num_envs, n_feet, dtype=torch.float, device=self.device,
            requires_grad=False)

        # [P3-f] 2026-09-02 hip_yaw 좌우 대칭용.
        # 사용자가 뷰어에서 "로봇의 왼다리가 더 바깥으로 돌아간다"고 관찰 -> 실측 확인:
        #   model_1000, cmd 0.4, 64/64 생존
        #   L_hip_yaw DC +22.13도 / R_hip_yaw DC -13.59도 -> 미러 잔차 +8.53도
        #   발 heading도 L +23.10 / R -13.05로 관절각을 거의 그대로 따라감
        #   (hip_yaw는 다리를 수직축으로 돌리므로 운동학적 잉여로 상쇄되지 않는다.
        #    어제 hip/ankle 비대칭이 발 자세에서 상쇄됐던 것과 다른 축이다.)
        # URDF상 두 hip_yaw는 axis=(0,0,1)로 동일 -> 시상면 반사 대칭이면 부호가 반대,
        # 즉 L+R = 0 이 대칭 조건이다(그래서 '차이'가 아니라 '합'을 벌한다).
        self.hip_yaw_dofs = [self.dof_names.index("L_hip_yaw_joint"),
                             self.dof_names.index("R_hip_yaw_joint")]
        self.hip_yaw_ema = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device,
            requires_grad=False)

        # hip_roll도 같은 구조. URDF상 좌우 모두 axis=(1,0,0)이고 범위만 미러
        # (R [-2.4, 0], L [0, 2.4]), 기본값도 R -0.03 / L +0.03으로 합이 0이다.
        # 따라서 hip_yaw와 동일하게 L + R = 0 이 대칭 조건이다.
        self.hip_roll_dofs = [self.dof_names.index("L_hip_roll_joint"),
                              self.dof_names.index("R_hip_roll_joint")]
        self.hip_roll_ema = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device,
            requires_grad=False)

        # ─────────────────────────────────────────────────────────────────
        # [P3-i] 2026-09-02 다리 6관절 전체 미러 대칭.
        # 배경: hip_roll 하나만 개별 항으로 잡았더니 나머지 관절이 무너졌다
        # (사용자 관찰: "너무 hip_roll에만 집중해서 다른게 무너졌어").
        # 관절마다 항을 따로 붙이는 방식 자체가 문제였으므로 6쌍을 한 항으로 다룬다.
        #
        # 미러 부호를 URDF 축에서 직접 유도했다. 시상면 반사에서 각속도는 유사벡터라
        # 부호가 (wx, wy, wz) -> (-wx, +wy, -wz)로 간다. 즉 pitch(y)는 유지, roll(x)와
        # yaw(z)는 반전. 좌우 관절 축이 서로 반대이면 거기서 한 번 더 반전된다.
        #   hip_pitch   R(0,1,0)  L(0,1,0)   같음 -> pitch 유지          -> +1 (L=R)
        #   hip_roll    R(1,0,0)  L(1,0,0)   같음 -> roll 반전           -> -1 (L=-R)
        #   hip_yaw     R(0,0,1)  L(0,0,1)   같음 -> yaw 반전            -> -1
        #   knee_pitch  R(0,-1,0) L(0,-1,0)  같음 -> pitch 유지          -> +1
        #   ankle_pitch R(0,-1,0) L(0,+1,0)  반대 -> pitch 유지 후 반전  -> -1  ⚠️
        #   ankle_roll  R(1,0,0)  L(1,0,0)   같음 -> roll 반전           -> -1
        #
        # ⚠️ 친구 원본 _reward_symmetry의 마스크 [1,-1,-1,1,+1,-1]은 ankle_pitch를 +1로
        # 두는데, 이 URDF에서는 -1이 맞다(위 유도). 어제 joint_asym.py가 ankle_pitch를
        # 27도로 잘못 냈다가 축 보정 후 8.9도가 나온 것도 -1을 적용한 결과라 실측과
        # 일치한다. 원본이 scale 0.0으로 꺼져 있는 이유가 위상 문제만이 아닐 수 있다.
        pair_names = [("L_hip_pitch_joint", "R_hip_pitch_joint", 1.0),
                      ("L_hip_roll_joint", "R_hip_roll_joint", -1.0),
                      ("L_hip_yaw_joint", "R_hip_yaw_joint", -1.0),
                      ("L_knee_pitch_joint", "R_knee_pitch_joint", 1.0),
                      ("L_ankle_pitch_joint", "R_ankle_pitch_joint", -1.0),
                      ("L_ankle_roll_joint", "R_ankle_roll_joint", -1.0)]
        self.sym_l_dofs = [self.dof_names.index(l) for l, _, _ in pair_names]
        self.sym_r_dofs = [self.dof_names.index(r) for _, r, _ in pair_names]
        self.sym_mask = torch.tensor(
            [m for _, _, m in pair_names], dtype=torch.float, device=self.device)
        # [P3-l] 2026-09-03 좌우를 '따로' 평균낸다. 예전에는 잔차(L - mask*R)를 하나로
        # EMA했는데, EMA는 선형이라 그 값은 아래 두 EMA의 차이와 수학적으로 동일하다.
        # 따로 두는 이유는 진폭(AC)을 좌우 비교하려면 각 관절의 '자기 평균'이 필요해서다.
        n_pair = len(pair_names)
        z = lambda: torch.zeros(self.num_envs, n_pair, dtype=torch.float,
                                device=self.device, requires_grad=False)
        self.leg_l_ema, self.leg_r_ema = z(), z()      # 좌/우 관절각의 시간평균 (DC)
        # 평균 대비 |편차|의 EMA = 보행 중 그 관절이 실제로 흔들리는 크기 (AC).
        # 이게 필요한 이유(2026-09-03 실측): joint_symmetry는 DC만 비교하므로
        # ankle_pitch가 L AC 3.39도 / R AC 6.98도로 2배 차이나는데도 DC 잔차는
        # -2.08도로 '대칭'으로 읽혔다. 즉 한쪽 발목이 거의 안 움직이는 것을 못 잡는다.
        # 물리적으로는 왼발이 발목을 못 굽혀 뒤꿈치로 찍는 것으로 나타났다.
        self.leg_l_amp, self.leg_r_amp = z(), z()
        # 평균 EMA는 0이 아니라 기본자세로 시작한다. 0에서 시작하면 리셋 직후
        # |편차|가 실제 진폭(약 0.06rad)이 아니라 기본자세 크기(약 0.2rad)로 잡혀
        # 진폭 EMA가 몇 초간 부풀려진다. 리셋 시 관절은 기본자세에 놓이므로
        # 기본자세로 초기화하는 것이 실제값에 가장 가깝다.
        self.leg_l_default = self.default_dof_pos[:, self.sym_l_dofs].clone()
        self.leg_r_default = self.default_dof_pos[:, self.sym_r_dofs].clone()
        self.leg_l_ema += self.leg_l_default
        self.leg_r_ema += self.leg_r_default
        # 왼쪽 진폭의 목표값(rad). 오른쪽 실측 진폭에서 뽑은 상수다. 상수인 것이 핵심 —
        # 식에 R이 들어가면 정책이 '오른쪽을 줄여서' 대칭을 맞출 수 있기 때문이다.
        # [P3-m] 2026-09-03 왼발 접지 자세 교정용 인덱스.
        # feet_indices 순서를 추측하지 않고 이름으로 찾는다(DOF 순서는 L이 먼저인데
        # URDF 파일 기재 순서는 R이 먼저라 좌우를 뒤바꾸기 쉽다).
        _bn = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        _fnames = [_bn[i] for i in self.feet_indices.tolist()]
        self.left_foot_slot = _fnames.index("L_foot_link")
        self.left_foot_body = int(self.feet_indices[self.left_foot_slot])
        self.right_foot_slot = _fnames.index("R_foot_link")
        self.right_foot_body = int(self.feet_indices[self.right_foot_slot])

        self.amp_target_left = torch.tensor(
            self.cfg.rewards.amp_target_left, dtype=torch.float, device=self.device)
        assert self.amp_target_left.numel() == n_pair, (
            f"amp_target_left 길이 {self.amp_target_left.numel()} != 관절쌍 {n_pair}")

        self._prepare_reward_function()
        self.init_done = True

        self._print_robot_indices()

    def _print_robot_indices(self):
        print("=" * 70)
        print("[QUBFlat] Robot index verification")
        print("-" * 70)
        print("DOF order:")
        for i, name in enumerate(self.dof_names):
            print(f"  [{i:2d}] {name}")
        print("-" * 70)
        print(f"feet_indices              : {self.feet_indices.cpu().tolist()}")
        if hasattr(self, "penalised_contact_indices"):
            print(f"penalised_contact_indices : {self.penalised_contact_indices.cpu().tolist()}")
        if hasattr(self, "termination_contact_indices"):
            print(f"termination_contact_indices: {self.termination_contact_indices.cpu().tolist()}")
        ankle_idx = [i for i, n in enumerate(self.dof_names) if "ankle" in n]
        print(f"ankle DOF indices (auto)  : {ankle_idx}")
        print("=" * 70)

    def step(self, actions):
        self._action_clip(actions)
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

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,
            self.commands[:, :3] * self.commands_scale,
            self.critic_obs_buf
        )

    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = (
            self.command_ranges["lin_vel_x"][env_ids, 1]
            - self.command_ranges["lin_vel_x"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges["lin_vel_x"][env_ids, 0]
        self.commands[env_ids, 1] = (
            self.command_ranges["lin_vel_y"][env_ids, 1]
            - self.command_ranges["lin_vel_y"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges["lin_vel_y"][env_ids, 0]
        self.commands[env_ids, 2] = (
            self.command_ranges["ang_vel_yaw"][env_ids, 1]
            - self.command_ranges["ang_vel_yaw"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges["ang_vel_yaw"][env_ids, 0]
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
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        n = self.cfg.env.num_actions

        idx = 0
        noise_vec[idx:idx + 3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        idx += 3
        noise_vec[idx:idx + 3] = noise_scales.gravity * noise_level
        idx += 3
        noise_vec[idx:idx + n] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        idx += n
        noise_vec[idx:idx + n] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel

        return noise_vec

    def _process_dof_props(self, props, env_id):
        """soft joint limit이 '기본자세를 포함하도록' 보정한다.

        왜 필요한가 (2026-09-02 실측으로 확인된 실제 버그):
          base_task는 soft limit을 관절 '범위의 중점' 기준으로 잡는다.
              m = (lower + upper) / 2,  soft = m +- 0.5 * r * soft_dof_pos_limit
          이 공식은 범위가 0을 중심으로 대칭인 관절에서만 말이 된다. 그런데 QUB의
          hip_roll은 R [-2.4, 0] / L [0, +2.4]로 극단적으로 치우쳐 있어서 중점이
          ∓1.2rad(∓69도)라는 엉뚱한 곳에 온다. soft_dof_pos_limit=0.9면
              R soft = [-2.28, -0.12],  L soft = [+0.12, +2.28]
          가 되어 기본자세(R -0.03 / L +0.03)조차 범위 밖이다.

        측정된 결과:
          - _reward_dof_pos_limits는 선형이라 R 하나가 매 스텝 0.1195의 위반을 낸다.
            학습 로그의 rew_dof_pos_limits = -0.134 중 89%가 이 관절 하나였다.
          - 이 항은 양 hip_roll을 6.9도 이상 '벌리라'고 요구하는데, 그렇게 하면
            발 간격이 약 0.41m가 되어 feet_distance_max(0.35, scale -3.0)에 0.18이
            걸린다. 0.18 > 0.12이므로 정책의 최적해는 '한쪽만 벌리고 다른 쪽은
            페널티를 그냥 내는 것'이 되고, 그것이 곧 절뚝임이었다.
          즉 두 항이 동시에 만족 불가능한 요구를 하고 있었고, 대칭 보상을 아무리
          더해도 이 충돌 위에 요구를 하나 더 얹는 것에 불과했다.

        수정: 중점 공식은 그대로 두되, 계산된 soft 범위가 기본자세를 배제하면
        그 경계를 기본자세 바깥(단, URDF 하드 리밋 안)까지 물러나게 한다.
        범위가 원래 0 대칭인 관절은 기본자세가 이미 안쪽이라 아무 영향이 없다.
        """
        props = super()._process_dof_props(props, env_id)
        if env_id != 0:
            return props

        # [P3-j] 기본값 False. 이 보정을 켜면 hip_roll을 바깥으로 밀던 힘이 사라지는데,
        # 방침이 '관절 사용 억제'가 아니라 '대칭 강제'로 정리되면서 끄기로 했다.
        # dof_pos_limits가 양다리를 벌리라고 요구하는 것 자체는 문제가 아니고,
        # 한쪽만 벌려서 절뚝이던 것이 문제였으며 그건 joint_symmetry가 막는다.
        # 자세한 비용 비교는 qub_flat_config.py의 fix_soft_limit_to_include_default 주석.
        if not getattr(self.cfg.rewards, "fix_soft_limit_to_include_default", False):
            return props

        margin = self.cfg.rewards.soft_limit_default_margin
        defaults = self.cfg.init_state.default_joint_angles
        for i, name in enumerate(self.dof_names):
            d = defaults.get(name, 0.0)
            hard_lo = props["lower"][i].item()
            hard_hi = props["upper"][i].item()
            lo = self.dof_pos_limits[i, 0].item()
            hi = self.dof_pos_limits[i, 1].item()
            # 기본자세가 soft 범위 밖이면 그쪽 경계를 물린다. 하드 리밋은 절대 안 넘는다.
            new_lo = max(hard_lo, min(lo, d - margin))
            new_hi = min(hard_hi, max(hi, d + margin))
            if new_lo != lo or new_hi != hi:
                print(f"[QUBFlat] soft limit 보정 {name}: "
                      f"[{lo:+.3f}, {hi:+.3f}] -> [{new_lo:+.3f}, {new_hi:+.3f}] "
                      f"(default {d:+.3f})")
            self.dof_pos_limits[i, 0] = new_lo
            self.dof_pos_limits[i, 1] = new_hi
        return props

    def _foot_x_pelvis(self):
        """양 발의 '골반 기준' 전후(x) 좌표. (num_envs, num_feet)

        base_link가 아니라 pelvis_link 기준으로 재는 이유:
          다리는 pelvis_link에 달려 있는데 그 사이에 torso_yaw_joint가 있다. base 기준으로
          재면 골반을 비트는 것만으로 좌우 오프셋이 전후축으로 섞여 수치가 좋아진다.
          실측으로 그 은폐량이 1.30cm(전체 7.02cm 중 19%)였다. 다리 자체의 대칭을
          보고 싶으면 다리가 달린 프레임에서 재야 한다.
        """
        rb = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)
        pelvis_pos = rb[:, self.pelvis_body_idx, 0:3]
        pelvis_quat = rb[:, self.pelvis_body_idx, 3:7]
        rel = self.foot_positions - pelvis_pos.unsqueeze(1)      # (N, feet, 3) 월드
        n_feet = rel.shape[1]
        # quat_rotate_inverse는 q와 v의 배치 차원이 같아야 해서 (envs, feet)를 펴서 계산
        q = pelvis_quat.repeat_interleave(n_feet, dim=0)
        return quat_rotate_inverse(
            q, rel.reshape(-1, 3)).reshape(self.num_envs, n_feet, 3)[:, :, 0]

    def _post_physics_step_callback(self):
        """[P3-e] 발의 착지/이지 이벤트를 추적한다.

        여기에 두는 이유: base_task.post_physics_step에서 이 콜백은 compute_foot_state()
        직후(= foot_positions/contact_forces가 갱신된 뒤), compute_reward() 직전에
        불린다. 보상 함수들이 읽기 전에 갱신되어야 하므로 이 자리가 맞다.

        착지 시점의 전후위치와 스윙 거리(착지 - 이지)를 발별로 EMA로 쌓아두면,
        '어디에 딛는가'와 '얼마나 크게 내딛는가'를 좌우로 직접 비교할 수 있다.
        스텝마다의 평균위치(foot_x_ema)와 달리 스윙/스탠스 프로파일이 좌우로 달라도
        구분해낼 수 있다.
        """
        super()._post_physics_step_callback()

        contacts = self.contact_forces[:, self.feet_indices, 2] > 1.0
        foot_x = self._foot_x_pelvis()
        a = self.event_ema_alpha

        touchdown = contacts & (~self.last_foot_contacts)
        liftoff = (~contacts) & self.last_foot_contacts

        # 스윙 거리 = 착지 위치 - 직전 이지 위치 (골반 기준 전후)
        stride = foot_x - self.foot_liftoff_x

        self.foot_touchdown_x_ema = torch.where(
            touchdown,
            (1.0 - a) * self.foot_touchdown_x_ema + a * foot_x,
            self.foot_touchdown_x_ema)
        self.foot_stride_ema = torch.where(
            touchdown,
            (1.0 - a) * self.foot_stride_ema + a * stride,
            self.foot_stride_ema)
        self.foot_liftoff_x = torch.where(liftoff, foot_x, self.foot_liftoff_x)
        self.last_foot_contacts = contacts

        # [P3-l] 다리 관절 좌/우 EMA 갱신. 리워드 함수 안이 아니라 여기서 하는 이유:
        # joint_symmetry(DC)와 joint_amplitude_symmetry(AC)가 같은 EMA를 공유하는데,
        # 각자 갱신하면 한 스텝에 두 번 갱신되어 시정수가 절반이 된다.
        a = self.step_sym_alpha
        l = self.dof_pos[:, self.sym_l_dofs]
        r = self.dof_pos[:, self.sym_r_dofs]
        # 진폭은 '갱신 전' 평균 기준으로 잰다. 같은 스텝의 평균으로 재면 방금 그 값이
        # 평균에 섞여 들어가 편차가 과소평가된다.
        self.leg_l_amp = (1.0 - a) * self.leg_l_amp + a * (l - self.leg_l_ema).abs()
        self.leg_r_amp = (1.0 - a) * self.leg_r_amp + a * (r - self.leg_r_ema).abs()
        self.leg_l_ema = (1.0 - a) * self.leg_l_ema + a * l
        self.leg_r_ema = (1.0 - a) * self.leg_r_ema + a * r

    def reset_idx(self, env_ids):
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
        # EMA를 0에서 다시 쌓게 함. 리셋 직후엔 좌우가 같은 자세라 차이도 0이고,
        # 시정수 2초 안에 실제값으로 수렴하므로 20초 에피소드에선 영향이 작다.
        self.foot_x_ema[env_ids] = 0.0
        # [P3-e] 골반 비틀림 / 착지위치 / 보폭 EMA도 같이 초기화.
        # 리셋 직후엔 좌우가 같은 자세라 차이가 0이므로 초반에는 페널티가 과소평가된다.
        # 이벤트 EMA(시정수 약 6초)는 20초 에피소드의 앞 1/3 정도가 워밍업인 셈이다.
        self.torso_yaw_ema[env_ids] = 0.0
        self.hip_yaw_ema[env_ids] = 0.0
        self.hip_roll_ema[env_ids] = 0.0
        # [P3-l] 평균 EMA는 기본자세로, 진폭 EMA는 0으로 되돌린다. 평균을 0으로 두면
        # 리셋 직후 |편차|가 부풀려져 진폭 항이 헛불을 낸다(__init__ 주석 참고).
        self.leg_l_ema[env_ids] = self.leg_l_default[0]
        self.leg_r_ema[env_ids] = self.leg_r_default[0]
        self.leg_l_amp[env_ids] = 0.0
        self.leg_r_amp[env_ids] = 0.0
        self.last_foot_contacts[env_ids] = False
        self.foot_touchdown_x_ema[env_ids] = 0.0
        self.foot_liftoff_x[env_ids] = 0.0
        self.foot_stride_ema[env_ids] = 0.0
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

    def _get_ankle_dof_indices(self):
        if not hasattr(self, "_ankle_dof_indices_cached"):
            ankle_idx = [i for i, name in enumerate(self.dof_names) if "ankle" in name]
            self._ankle_dof_indices_cached = torch.tensor(
                ankle_idx, dtype=torch.long, device=self.device
            )
        return self._ankle_dof_indices_cached

    # ===================== reward functions =====================

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

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
        """v6: 자세 의존 alive bonus (Anti-Lying-down)
        	제대로 서있을 때만 +1, 누워있거나 squat이면 0을 부여하여 보상 해킹 원천 차단"""
        base_height = self.root_states[:, 2]
        
        # 1. 고도가 0.55m 이상일 것 (주저앉기 방지)
        height_ok = (base_height > 0.55).float()
        
        # 2. 몸통이 수평을 유지할 것 (약 32도 이내의 기울기, 눕기 방지)
        upright_ok = (self.projected_gravity[:, 2] < -0.85).float()
        
        # 두 조건을 모두 만족(AND 연산)할 때만 1.0 반환
        return height_ok * upright_ok

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

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
        """다리 너무 가까워질 때 페널티 (min_feet_distance 미만)"""
        feet_distance = torch.norm(self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1)
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1)
        return reward

    def _reward_feet_distance_max(self):
        """[v5 신규] 다리 너무 벌어질 때 페널티 (max_feet_distance 초과).
        주저앉기 reward hacking 방지의 핵심.
        정상 stance ~0.25m, 0.35m 넘으면 페널티 시작."""
        feet_distance = torch.norm(
            self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1
        )
        return torch.clip(feet_distance - self.cfg.rewards.max_feet_distance, 0, 2)

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

    def _reward_posture(self):
        """default pose에서 벗어나는 것을 페널티. v5에서 활성화 (anti-squat 핵심)."""
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_ankle_regularization(self):
        ankle_idx = self._get_ankle_dof_indices()
        return torch.sum(
            torch.square(self.dof_pos[:, ankle_idx] - self.default_dof_pos[:, ankle_idx]),
            dim=1,
        )

    def _reward_ang_vel_z(self):
        return torch.square(self.base_ang_vel[:, 2] - self.commands[:, 2])

    def _reward_yaw_drift(self):
        roll, pitch, yaw = get_euler_xyz(self.base_quat)
        yaw_wrapped = torch.atan2(torch.sin(yaw), torch.cos(yaw))
        return torch.square(yaw_wrapped)

    def _reward_step_symmetry(self):
        """좌우 보폭 비대칭 페널티 — 한쪽 발만 계속 앞에 딛는 것을 막는다.

        왜 순간 좌우 비교(_reward_symmetry)를 안 쓰는가:
          보행은 좌우가 180도 위상차라 '매 순간 다른 것'이 정상이다. 같은 시점의 좌우를
          비교해 벌점을 주면 양발을 함께 움직이는 제자리뛰기 쪽으로 밀려난다.
          (친구 원본에서 _reward_symmetry가 scale 0.0으로 꺼져 있는 이유로 보임)

        그래서 보행 주기보다 긴 시정수로 평균 낸 '몸 기준 전후 위치'를 좌우 비교한다.
        대칭 보행이면 두 평균이 같아지고, 한쪽을 더 앞에 딛으면 차이가 남는다.

        2026-09-02 실측(Sep02_00-24-10_/model_5000, cmd 0.4, 10초):
          발0 평균 +0.0299m / 발1 평균 -0.0640m  -> 차이 9.4cm
          특히 발1은 최대 전방 위치가 -0.0023m로 몸 앞으로 아예 나오지 못했음(끌려감).

        🔴 [P3-e] 2026-09-02 기준 프레임을 base_link -> pelvis_link로 변경.
          base 기준으로 재면 torso_yaw로 골반을 비트는 것만으로 좌우 오프셋이 전후축에
          섞여 수치가 좋아진다. 실제로 그러고 있었다(model_2000 실측: base 5.72cm vs
          pelvis 7.02cm, 1.30cm가 은폐되고 있었음). 다리는 pelvis에 달려 있으므로
          다리의 대칭은 pelvis 기준으로 재는 것이 맞다. 자세한 근거는 __init__ 참고.
        """
        foot_x = self._foot_x_pelvis()

        # compute_reward가 스텝당 한 번만 호출하므로 여기서 EMA를 갱신해도 안전하다.
        self.foot_x_ema = ((1.0 - self.step_sym_alpha) * self.foot_x_ema
                           + self.step_sym_alpha * foot_x)
        return torch.square(self.foot_x_ema[:, 0] - self.foot_x_ema[:, 1])

    def _reward_torso_yaw_drift(self):
        """골반이 한쪽으로 '틀어진 채 고정'되는 것만 벌한다(진행방향 정렬).

        보행 중 골반이 좌우로 돌아가는 것(AC)은 사람도 하는 정상 동작이라 살려야 한다.
        이상한 것은 한쪽으로 치우친 정적 성분(DC)이다. 그래서 관절각 자체가 아니라
        시정수 2초(=3.4 보행주기) EMA를 벌한다. 1.7Hz 진동은 EMA에서
        1/sqrt(1+(w*tau)^2) ~ 0.047배로 눌리므로, 실측 +-1.9도 AC는 +-0.09도만 남고
        DC +3.40도는 거의 그대로 통과한다. 즉 AC는 건드리지 않고 DC만 잡는다.

        torso_yaw_joint는 base_link -> pelvis_link 관절이므로, base가 명령 방향을
        향하고 있는 한(tracking_ang_vel이 담당) 이 EMA가 0이면 '골반도 평균적으로
        진행 방향을 향한다'는 뜻이 된다.

        ⚠️ 이 +3.40도 비틀림은 다리 비대칭이 만드는 요 모멘트에 대한 '보상 동작'일
        가능성이 있다. 원인(다리 비대칭)이 남은 채 이것만 강하게 누르면 로봇이 방향을
        트는 쪽으로 갈 수 있으므로, step/landing/stride 대칭 항들과 반드시 같이 쓴다.
        """
        tyaw = self.dof_pos[:, self.torso_yaw_dof]
        self.torso_yaw_ema = ((1.0 - self.step_sym_alpha) * self.torso_yaw_ema
                              + self.step_sym_alpha * tyaw)
        return torch.square(self.torso_yaw_ema)

    def _reward_hip_yaw_symmetry(self):
        """좌우 hip_yaw를 대칭으로 — 한 다리만 더 바깥으로 돌아가는 절뚝임을 막는다.

        왜 순간값 비교가 아니라 EMA인가:
          보행은 좌우 180도 위상차라 '매 순간 좌우가 다른 것'이 정상이다. 순간 미러
          잔차를 벌하면 기존 _reward_symmetry가 scale 0.0으로 꺼져 있는 것과 똑같은
          덫(양발이 함께 움직이는 깡충뛰기)에 빠진다. 그래서 보행 주기보다 긴 시정수
          (2초 = 3.4주기)로 평균 낸 뒤 미러 잔차를 본다. 위상은 평균이 지운다.

        왜 '차이'가 아니라 '합'인가:
          URDF상 두 hip_yaw는 axis=(0,0,1)로 동일하다. 시상면 반사 대칭에서 요는 부호가
          뒤집히므로 대칭 조건은 L = -R, 즉 L + R = 0 이다. (친구 원본의 미러 마스크
          [1,-1,-1,1,1,-1]에서 hip_yaw가 -1인 것과 같은 이야기.)
          L - R을 벌하면 오히려 정상적인 대칭 자세를 깨뜨린다.

        ⚠️ 이 항은 '좌우 균형'만 잡는다. 양발이 똑같이 바깥으로 벌어진 대칭적 팔자걸음
        (실측상 공통 성분이 약 18도)은 L+R=0을 만족하므로 이 항으로는 줄지 않는다.
        그건 별개의 문제이고, 필요하면 hip_yaw 크기 자체를 벌하는 항을 따로 둬야 한다.
        """
        hy = self.dof_pos[:, self.hip_yaw_dofs]
        self.hip_yaw_ema = ((1.0 - self.step_sym_alpha) * self.hip_yaw_ema
                            + self.step_sym_alpha * hy)
        return torch.square(self.hip_yaw_ema[:, 0] + self.hip_yaw_ema[:, 1])

    def _reward_joint_symmetry(self):
        """다리 6관절 전체의 좌우 미러 대칭. hip_yaw/hip_roll 개별 항을 대체한다.

        왜 하나로 합쳤는가: 관절마다 항을 따로 붙이면 한 관절에 압력이 몰려 나머지가
        무너진다(2026-09-02 실측: hip_roll 잔차 6.58->0.88도로 잡히는 동안 앞뒤 발
        위치가 3.69->5.50cm로 악화). 6쌍을 한 항으로 묶으면 어느 관절이 어긋나든
        같은 크기로 벌하므로 그런 편중이 생기지 않는다.

        왜 EMA인가: 보행은 좌우 180도 위상차라 매 순간 좌우가 다른 것이 정상이다.
        순간 미러 비교는 양발이 함께 움직이는 깡충뛰기를 유도한다(친구 원본
        _reward_symmetry가 꺼져 있는 이유). 시정수 2초(3.4 보행주기) 평균을 비교하면
        위상은 상쇄되고 '한쪽으로 치우쳐 고정된 성분'만 남는다.

        잔차 정의: mask=+1이면 대칭 조건이 L=R, mask=-1이면 L=-R이므로
        어느 경우든 잔차 = ema_L - mask * ema_R 로 통일된다. 미러 부호의 URDF 유도는
        __init__ 주석 참고.
        """
        # EMA 갱신은 _post_physics_step_callback에서 한 번만 한다(아래 진폭 항과 공유).
        # EMA가 선형이라 '잔차의 EMA'와 '각 EMA의 잔차'는 수학적으로 동일하다.
        residual = self.leg_l_ema - self.sym_mask.unsqueeze(0) * self.leg_r_ema
        return torch.sum(torch.square(residual), dim=1)

    def _reward_left_foot_flat_contact(self):
        """왼발이 땅에 닿아 있는 동안 '발바닥 전체'가 땅에 붙게 한다.

        측정 방법: 발 좌표계에서 본 중력 방향. 발바닥이 땅과 평행하면 [0,0,-1]이고,
        x/y 성분이 그대로 기울기가 된다(base의 projected_gravity와 같은 개념).
        x는 앞뒤(발끝/뒤꿈치 들림), y는 좌우(안/바깥 모서리만 닿음) 기울기다.

        2026-09-03 실측(Sep03_19-41-38_/model_3500, cmd 0.4):
            L 스탠스 중  x -7.00도  y -6.34도  -> 전체 기울기 9.71도
            R 스탠스 중  x -0.13도  y +1.05도  -> 전체 기울기 1.11도
          오른발은 사실상 평평하게 붙는데 왼발만 9배 기울어 있다.

        ⚠️ 이전 항(_reward_left_foot_contact_pitch)이 틀린 양을 겨냥했던 이유:
          1) roll을 아예 보지 않았다. 실측 기울기의 절반이 좌우 성분(-6.34도)이었다.
          2) 기준이 골반이었다. 보행 중 골반이 앞으로 기울어 있어서 골반 기준 pitch가
             -1.32도(평평해 보임)여도 땅 기준으로는 -7.00도였다.
          '땅에 닿는가'는 월드 기준 문제이므로 중력으로 재는 것이 맞다.

        접촉 중에만 평가한다. 스윙 중에는 발을 들어 지면을 피해야 하므로 평평할 이유가
        없고, 그때까지 평평하게 만들면 발이 걸린다.

        오른발은 식에 없다(사용자: "오른발은 충분해"). 목표도 상대값이 아니라 0
        (=완전히 평평)이라, 오른발을 나쁘게 만들어 만족시킬 방법이 구조적으로 없다.
        오른발이 1.11도를 달성하고 있다는 것이 0이 도달 가능한 목표라는 증거다.
        """
        return self._foot_flat_penalty(self.left_foot_body)

    def _foot_flat_penalty(self, body_idx):
        """한 발의 '접촉 중 발바닥 기울기' 벌점. 좌우 항이 공유하는 계산.

        발 좌표계에서 본 중력의 x,y 성분이 곧 기울기다(발바닥이 땅과 평행하면 [0,0,-1]).
        x는 앞뒤(발끝/뒤꿈치 들림), y는 좌우(안/바깥 모서리만 닿음)를 잡는다.
        목표가 상대값이 아니라 절대 0이므로, 좌우 항을 둘 다 켜도 한쪽이 다른 쪽을
        나쁘게 만들어 자기 항을 만족시키는 경로가 생기지 않는다.
        """
        rb = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)
        proj_g = quat_rotate_inverse(rb[:, body_idx, 3:7], self.gravity_vec)
        in_contact = (self.contact_forces[:, body_idx, 2] > 1.0).float()
        return in_contact * torch.sum(torch.square(proj_g[:, :2]), dim=1)

    def _reward_right_foot_flat_contact(self):
        """오른발도 발바닥 전체가 땅에 닿게 한다. 왼발 항과 완전히 같은 형태.

        2026-09-03 실측(Sep03_20-50-02_/model_1500, 스탠스 중):
            R  x -0.06도  y +3.23도  -> 전체 3.24도
          기울기가 거의 전부 좌우 성분이다. 즉 앞뒤로는 평평한데 옆으로 돌아가
          안쪽 모서리로 딛고 있다(사용자 관찰: "오른발 발 안쪽으로 걷는구나").

        왼발 항을 넣기 전 오른발은 0.87도였는데, 왼발이 9.93 -> 0.22도로 잡히는 동안
        오른발이 3.24도로 밀려났다. 오른발은 어떤 리워드 수식에도 없었으므로 직접적인
        압력의 결과가 아니라 보행 재최적화의 부수 효과로 보이고, 이제 명시적으로 잡는다.

        스케일을 왼발(-15)보다 크게 잡은 이유: 제곱항이라 오차가 작을수록 압력도
        제곱으로 작아진다. 오른발 오차(3.24도)는 왼발 시작점(9.93도)의 1/3이라
        같은 스케일이면 압력이 1/8.5로 떨어져 사실상 놀게 된다.
        """
        return self._foot_flat_penalty(self.right_foot_body)

    def _reward_left_foot_contact_pitch(self):
        """왼발이 '땅에 닿아 있는 동안' 발 피치를 오른발 수준으로 맞춘다.

        2026-09-03 실측(Sep03_18-12-02_/model_5000, cmd 0.4, 골반 기준 발 피치):
            접지 순간   L -7.88도 (sd 3.28)  /  R +1.19도 (sd 1.00)
            스탠스 중   L -7.51도            /  R +2.10도
          오른발은 거의 평평하게 닿는데 왼발은 9도 젖혀진 채 닿고, 스탠스 내내
          그 각도를 유지한다. 이것이 화면에서 '왼발이 뒤꿈치로만 디딘다'로 보인다.

        왜 관절이 아니라 발을 겨냥하는가:
          같은 측정에서 왼발 피치의 '범위'는 18.2도로 오른발(6.2도)보다 3배 컸다.
          반면 ankle_pitch 관절 진폭은 왼쪽이 더 작았다(3.69 vs 6.48도). 즉 왼발
          피치는 발목이 아니라 다리 전체 자세에서 나온다. 관절을 겨냥하면 표적이
          어긋난다. 문제의 정의가 '발이 어떻게 닿는가'이므로 발을 직접 겨냥한다.

        왜 왼발만 들어가는가:
          오른발은 이미 잘 걷고 있어 건드리면 손해다(사용자: "오른발 정말 완벽하니까
          건들지마"). 수식에 오른발이 없으므로 오른발을 나쁘게 만들어 이 항을
          만족시킬 방법이 구조적으로 없다. 목표값도 오른발 실측에서 뽑은 상수다.

        접촉 중에만 평가하는 이유: 스윙 중에는 발끝을 들어 지면을 피해야 하므로
        그때까지 평평하게 만들면 발이 걸린다. 벌점은 접지 구간에만 걸어야 한다.
        """
        rb = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)
        q_rel = quat_mul(quat_conjugate(rb[:, self.pelvis_body_idx, 3:7]),
                         rb[:, self.left_foot_body, 3:7])
        _, pitch, _ = get_euler_xyz(q_rel)
        pitch = wrap_to_pi(pitch)
        in_contact = (self.contact_forces[:, self.left_foot_body, 2] > 1.0).float()
        return in_contact * torch.square(
            pitch - self.cfg.rewards.left_foot_contact_pitch_target)

    def _reward_joint_amplitude_symmetry(self):
        """좌우 관절의 '움직이는 크기'를 맞춘다 — 한쪽만 굳는 것을 막는다.

        왜 필요한가 (joint_symmetry로는 못 잡는 구멍):
          joint_symmetry는 시간평균(DC)만 비교하므로 진폭에는 완전히 무관심하다.
          한쪽이 7도 흔들리고 다른 쪽이 3.4도 흔들려도 평균만 미러면 잔차가 0이다.
          2026-09-03 실측(Sep03_17-37-52_/model_1000)이 정확히 그 상태였다:
            ankle_pitch  L: DC -17.53 AC 3.39 / R: DC +15.45 AC 6.98
            -> DC 잔차는 -2.08도로 '대칭'인데 진폭은 2배 차이.
          물리적으로는 왼발이 발목을 충분히 굽히지 못해 뒤꿈치→발끝으로 구르지 못하고
          뒤꿈치로 찍는 것으로 나타났다(사용자 관찰: "왼발 ankle_pitch 고정된 것 같다,
          땅에 디딜 때 뒤꿈치 닿는다").

        진폭은 '평균 대비 |편차|'의 EMA로 잰다. 부호가 없으므로 미러 마스크와 무관하고,
        좌우 180도 위상차에도 영향받지 않는다(위상이 달라도 크기는 같아야 정상).
        분산 대신 |편차|를 쓰는 이유는 rad 단위라 다른 항들과 스케일 비교가 쉽고,
        제곱근이 없어 0 근처 기울기 문제도 없기 때문이다.

        ★ 왜 (L - R)^2이 아니라 '왼쪽만' 보는가 (2026-09-03, 사용자 지시):
          차이를 벌하면 정책은 '왼쪽을 키우기'와 '오른쪽을 줄이기' 중 싼 쪽을 고른다.
          오른발은 이미 잘 걷고 있으므로 줄어들면 손해다("오른발은 건드리지 말자").
          그래서 수식에서 R을 완전히 제거하고, 왼쪽 진폭이 '측정으로 고정한 목표값'에
          못 미칠 때만 벌한다. R이 식에 없으므로 오른쪽을 줄여서 이득 볼 방법이 없다.
          단방향(clip)이라 왼쪽이 목표보다 커지는 것도 벌하지 않는다 — 실측상 6개 중
          4개 관절은 이미 왼쪽이 더 크므로, 양방향이면 그것들을 깎아버린다.

        ⚠️ 목표값은 실측 상수다(config의 amp_target_left). 보행 양상이 크게 바뀌면
        의미가 없어지므로, 게이트가 걸려 있는지(=값이 0으로 붙는지) 확인하고 필요하면
        R을 다시 재서 갱신할 것.
        """
        deficit = torch.clip(self.amp_target_left.unsqueeze(0) - self.leg_l_amp,
                             min=0.0)
        return torch.sum(torch.square(deficit), dim=1)

    def _reward_hip_roll_symmetry(self):
        """좌우 hip_roll 대칭. _reward_hip_yaw_symmetry와 완전히 같은 논리다.

        URDF상 좌우 모두 axis=(1,0,0)이고 범위만 미러이므로 대칭 조건은 L + R = 0.
        (기본값 R -0.03 / L +0.03의 합이 0인 것이 이를 확인해 준다.)
        위상 때문에 순간값이 아니라 EMA를 쓰는 이유도 hip_yaw 쪽 주석과 동일하다.

        ⚠️ 이 항은 dof_pos_limits와 구조적으로 경쟁한다. _process_dof_props가 soft
        limit을 '범위 중점' 기준으로 계산하는데 hip_roll은 범위가 한쪽으로 치우쳐 있어
        중점이 ∓1.2rad(±69도)에 온다. 그 결과 soft 범위가 R [-2.28,-0.12] /
        L [+0.12,+2.28]이 되어 기본자세 ∓0.03조차 범위 밖이고, 가만히 서 있어도
        dof_pos_limits가 양다리를 바깥으로 민다. 좌우가 이 압력에 다르게 순응하면
        그 자체가 비대칭이 되므로, 이 항이 안 듣거든 스케일을 올리기 전에
        soft_dof_pos_limit 쪽을 먼저 의심할 것.
        """
        hr = self.dof_pos[:, self.hip_roll_dofs]
        self.hip_roll_ema = ((1.0 - self.step_sym_alpha) * self.hip_roll_ema
                             + self.step_sym_alpha * hr)
        return torch.square(self.hip_roll_ema[:, 0] + self.hip_roll_ema[:, 1])

    def _reward_foot_landing_symmetry(self):
        """좌우 발이 '딛는 위치'를 같게 한다.

        step_symmetry(스텝마다의 평균 위치)와 다른 점: 평균이 같아도 착지 지점이 다를 수
        있다(스윙/스탠스 프로파일이 좌우로 다르면). 사용자가 지목한 '발 떨어지는 위치'는
        평균이 아니라 착지 지점이므로 그 이벤트를 직접 잡아서 비교한다.
        EMA 갱신은 _post_physics_step_callback의 착지 전이에서만 일어난다.
        """
        return torch.square(
            self.foot_touchdown_x_ema[:, 0] - self.foot_touchdown_x_ema[:, 1])

    def _reward_stride_symmetry(self):
        """좌우 보폭(이지->착지 전진거리)을 같게 한다.

        ⚠️ model_2000 실측에서 보폭은 이미 거의 같았다(0.1481 vs 0.1488, 0.5% 차이).
        따라서 이 항은 지금 당장 뭔가를 고치는 항이 아니라, 다른 대칭 항들을 강하게
        누르는 과정에서 '평균은 맞췄지만 한쪽 보폭이 줄어드는' 식의 퇴행이 생기지 않게
        막는 보험이다. 값이 계속 0 근처면 정상이다.
        """
        return torch.square(
            self.foot_stride_ema[:, 0] - self.foot_stride_ema[:, 1])

    def _reward_symmetry(self):
        l_actions = self.actions[:, 1:7]
        r_actions = self.actions[:, 7:13]
        if not hasattr(self, "_sym_mirror"):
            self._sym_mirror = torch.tensor(
                [1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
                device=self.device, dtype=torch.float,
            )
        diff = l_actions - r_actions * self._sym_mirror
        return torch.sum(torch.square(diff), dim=1)
