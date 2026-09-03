# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym.torch_utils import *
from legged_gym.envs import *
from legged_gym.utils import (
    get_args,
    export_policy_as_jit,
    export_mlp_as_onnx,
    task_registry,
    Logger,
)

import numpy as np
import torch
import matplotlib.pyplot as plt


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.episode_length_s = 30
    # 100 -> 24: 뷰어는 눈으로 보는 용도라 개체가 많으면 화면이 빽빽해 한 대의 보행을
    # 따라가기 어렵다. 통계는 헤드리스 측정 스크립트(64 env)가 따로 내므로
    # 여기서 개체 수를 줄여도 잃는 것이 없다.
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 24)

    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 20
    env_cfg.terrain.terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
    env_cfg.terrain.max_init_terrain_level = 4
    env_cfg.terrain.curriculum = True
    env_cfg.noise.add_noise = True
    env_cfg.noise.noise_level = 0.5
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 3
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kd = False
    env_cfg.domain_rand.randomize_motor_torque = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    env_cfg.domain_rand.randomize_action_delay = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # get robot_type
    # get robot_type
    # get robot_type
    robot_type = os.getenv("ROBOT_TYPE")

    # 2026-09-01: 친구 config 이식 후 num_commands가 4->3으로 바뀌어 길이를 맞춤.
    # 값도 0.5 전진 -> 0.0으로 변경한 이유: 이 config는 zero_command_prob=0.0인데
    # _resample_commands의 판정이 `rand > zero_command_prob`라서 사실상 모든 env의
    # 커맨드가 0으로 세팅됨(= Phase 2 초반 "제자리에서 박자만 학습" 설계 그대로).
    # 즉 정책은 속도명령 0만 겪어봤으므로 0.5를 강제하면 학습분포 밖이라 무의미.
    # 이 단계의 판정 기준은 전진 속도가 아니라 "발이 리듬있게 들리는가".
    # 🔴 [P3] 2026-09-02: 0.0 -> 0.2. 제자리 단계에선 정책이 커맨드 0만 겪어봐서 0으로
    # 뒀었지만, zero_command_prob=0.8로 전진을 개방했으므로 이제 학습 범위 안의 전진
    # 명령으로 테스트해야 함. 0.2는 lin_vel_x 시작 범위(±0.2)의 상단 = 학습분포 안쪽.
    # (커리큘럼이 범위를 넓히면 이 값도 같이 올려서 확인할 것)
    # 🔴 [P3] 0.2 -> 0.4. iteration 2000 속도 스윕 실측 결과 반영:
    #   0.2/0.4/0.6 → 생존율 100%, 오차 13%/11%/10% (실제속도가 명령의 약 90%)
    #   0.7        → 생존율 17.7%로 붕괴,  0.8 이상 → 0%
    # 즉 실사용 한계는 0.6. play.py는 정지 상태에서 곧바로 명령을 주는(가속 구간 없는)
    # 조건이라 한계치인 0.6보다 여유 있는 0.4를 화면 확인용으로 사용.
    # 0.6 -> 0.5: 학습 커맨드 범위를 ±0.5로 바꿨으므로 학습분포 안쪽 값으로 맞춤.
    commands_val = to_torch(
        [0.5, 0.0, 0.0],
        device=env.device
    )

    action_scale = env.cfg.control.action_scale_pos if robot_type == "WF_TRON1A"\
        else env.cfg.control.action_scale

    obs, obs_history, commands, _ = env.get_observations()

    print("COMMANDS_VAL =", commands_val)
    print("COMMANDS_VAL SHAPE =", commands_val.shape)
    print("ENV COMMAND SHAPE =", commands.shape)
    
    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    # train_cfg.runner.checkpoint = -1

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    encoder = ppo_runner.get_inference_encoder(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            args.task,
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor,
            path,
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.encoder,
            path,
            "encoder",
            ppo_runner.alg.encoder.num_input_dim,
        )

    logger = Logger(env.dt)
    robot_index = 5  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    # 100 -> env.max_episode_length + 1: at 100 this used to call logger.plot_states(),
    # which opens a blocking matplotlib window (plt.show()) and freezes the sim loop
    # until someone closes it by hand. That breaks unattended/automated checkpoint checks.
    stop_state_log = int(env.max_episode_length) + 1
    stop_rew_log = (
        env.max_episode_length + 1
    )  # number of steps before print average episode rewards
    # camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    # camera_vel = np.array([1.0, 1.0, 0.0])
    # camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    est = None
    for i in range(10 * int(env.max_episode_length)):
        est = encoder(obs_history)
        actions = policy(torch.cat((est, obs, commands), dim=-1).detach())

        env.commands[:, :] = commands_val

        if i % 20 == 0:
            # 주기적 상태 로그. 목표(cmd)와 실제(act)를 나란히 찍는 이유: 예전 형식은
            # actual_vx만 찍어서 그 값이 좋은 건지 나쁜 건지 판단하려면 매번 config의
            # 명령 범위를 따로 확인해야 했다. 목표가 옆에 있어야 오차를 바로 읽는다.
            #
            # foot_lift: 두 발 중 더 높이 든 발의 높이(env 평균). 제자리 보행 단계에서는
            # 전진속도(act_vx)보다 이 값이 주기적으로 오르내리는지가 핵심 판정 기준.
            # tyaw: torso_yaw 관절각(도). base_link -> pelvis_link 관절이라 한쪽으로
            # 치우쳐 고정되면 골반을 비틀어 좌우 비대칭을 가리고 있다는 신호다.
            # alive: keep_balance와 같은 판정(높이 0.55 초과 & 뒤집히지 않음).
            # 형식: 열 표 대신 항목마다 라벨을 붙인다. 표 형식은 헤더가 화면 밖으로
            # 스크롤되면 어느 숫자가 무엇인지 알 수 없고, 실제로 WezTerm 89칸 패널에서
            # 마지막 열이 다음 줄로 넘어가 읽기 힘들었다.
            # 첫 줄만 봐도 속도 추종을 판단할 수 있게 목표/실제/오차를 첫 줄에 모으고,
            # 나머지 진단값은 들여쓴 둘째 줄로 내린다. 두 줄 다 85칸 안에 들어온다.
            # (한글 라벨은 터미널에서 글자당 2칸을 차지하므로 폭 계산에 주의)
            fh = env.foot_heights
            cmd_vx = env.commands[:, 0].mean().item()
            act_vx = env.base_lin_vel[:, 0].mean().item()
            # 목표가 0이면 상대오차가 정의되지 않으므로 표시를 생략한다.
            err = f"오차 {abs(act_vx - cmd_vx) / abs(cmd_vx) * 100:5.1f}%" \
                if abs(cmd_vx) > 1e-3 else "오차     -"
            alive = ((env.root_states[:, 2] > 0.55)
                     & (env.projected_gravity[:, 2] < -0.85))
            # tyaw: torso_yaw 관절각(도). base_link -> pelvis_link 관절이라 한쪽으로
            # 치우쳐 고정되면 골반을 비틀어 좌우 비대칭을 가리고 있다는 신호다.
            tyaw = np.rad2deg(env.dof_pos[:, env.torso_yaw_dof].mean().item()) \
                if hasattr(env, "torso_yaw_dof") else float("nan")
            # alive: keep_balance와 같은 판정(높이 0.55 초과 & 뒤집히지 않음).
            print(f"[{i * env.dt:6.1f}s] "
                  f"목표속도 {cmd_vx:+.3f} m/s  →  실제속도 {act_vx:+.3f} m/s   "
                  f"{err}   생존 {int(alive.sum())}/{env.num_envs}")
            print(f"          몸높이 {env.root_states[:, 2].mean().item():.3f} m   "
                  f"발높이 L {fh[:, 0].mean().item():.3f} R {fh[:, 1].mean().item():.3f} m   "
                  f"허리각 {tyaw:+.1f}°   "
                  f"요속도 {env.base_ang_vel[:, 2].mean().item():+.3f}")

        obs, rews, dones, infos, obs_history, commands, _ = env.step(
            actions.detach()
        )
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(
                    LEGGED_GYM_ROOT_DIR,
                    "logs",
                    train_cfg.runner.experiment_name,
                    "exported",
                    "frames",
                    f"{img_idx}.png",
                )
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_offset = np.array(env_cfg.viewer.pos)
            target_position = np.array(
                env.base_position[robot_index, :].to(device="cpu")
            )
            target_position[2] = 0
            camera_position = target_position + camera_offset
            # env.set_camera(camera_position, target_position)

        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": actions[robot_index, joint_index].item() * action_scale,
                    "dof_pos": (
                        env.dof_pos[robot_index, joint_index]
                        - env.raw_default_dof_pos[joint_index]
                    ).item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "command_x": env.commands[robot_index, 0].item(),
                    "command_y": env.commands[robot_index, 1].item(),
                    "command_yaw": env.commands[robot_index, 2].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "power": torch.sum(env.power[robot_index, :]).item(),
                    "contact_forces_z": env.contact_forces[
                        robot_index, env.feet_indices, 2
                    ]
                    .cpu()
                    .numpy(),
                }
            )
            # print(torch.sum(env.power[robot_index, :]).item())
            if est != None:
                logger.log_states(
                    {
                        "est_lin_vel_x": est[robot_index, 0].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_y": est[robot_index, 1].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_z": est[robot_index, 2].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                    }
                )
        elif i == stop_state_log:
            logger.plot_states()

        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == "__main__":
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    play(args)
