# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.base_config import BaseConfig
from legged_gym import LEGGED_GYM_ROOT_DIR

import os
robot_type = os.getenv("ROBOT_TYPE")

class QUBCfg(BaseConfig):
    class env:
        # 256 -> 512 -> 1024: 512에서도 GPU 여유 확인함(4.8GB/12GB 사용, 2026-08-29).
        # 🔴 4차 개선(round 5c): 512 → 1024. entropy_coef를 원본 수준(0.01)으로 올렸는데도
        # iteration 5000까지 actual_vx가 거의 0 근처에 머무는 정체가 안 풀림. must-follow
        # 원본인 pointfoot_flat_config.py와 대조해보니 num_envs=8192로 QUB보다 16배 많음
        # (다른 PPO 하이퍼파라미터는 전부 동일) — 이 논문/저장소의 핵심 아이디어 자체가
        # "병렬 env를 많이 굴려서 매 업데이트마다 다양한 경험(우연히 좋은 스텝 시퀀스를
        # 밟은 env 포함)을 확보하는 것"이라, entropy_coef보다 이게 더 근본적인 탐색 레버일
        # 가능성이 큼. 이 GPU(12GB)로는 8192는 무리라 계산상 안전한 1024로 올림.
        num_envs = 1024
        num_observations = 51
        num_critic_observations = 3 + num_observations
        num_height_samples = 117
        num_actions = 13
        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 20
        obs_history_length = 10
        dof_vel_use_pos_diff = True
        fail_to_terminal_time_s = 0.5

    class terrain:
        mesh_type = "plane"
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.8
        measure_heights = False
        critic_measure_heights = True
        measured_points_x = [
            -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
        ]
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5 + 4
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        slope_treshold = 0.75

    class commands:
        curriculum = True
        smooth_max_lin_vel_x = 2.0
        smooth_max_lin_vel_y = 1.0
        non_smooth_max_lin_vel_x = 1.0
        non_smooth_max_lin_vel_y = 1.0
        max_ang_vel_yaw = 3.0
        curriculum_threshold = 0.75
        num_commands = 3
        resampling_time = 5.0
        heading_command = True
        min_norm = 0.1
        zero_command_prob = 1.0

        class ranges:
            # ⚠️ zero_command_prob=1.0 이므로 아래 범위에서 항상 "0이 아닌" 명령이 샘플링됨
            # (zero_command_idx는 rand>zero_command_prob 조건이라 1.0일 때는 절대 걸리지 않음)
            lin_vel_x = [-0.5, 0.5]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.5, 0.5]
            heading = [0.0, 0.0]

    class gait:
        num_gait_params = 4
        resampling_time = 5
        
        # ========== 개선안 A: 자연스러운 제자리 걷기 (권장) ==========
        class ranges:
            # 🔴 원본: [1.0, 1.0] → 🟢 개선: [0.8, 1.2]
            # 설명: 다양한 보행 속도를 학습하게 함
            frequencies = [0.8, 1.2]
            
            # 🔴 원본: [0.5, 0.5] → 🟢 개선: [0.4, 0.6]
            # 설명: 완벽한 180도 교대가 아닌 약간의 변화도 허용
            offsets = [0.4, 0.6]
            
            # 🔴 원본: [0.6, 0.6] → 🟢 개선: [0.5, 0.7]
            # 설명: stance/swing 비율을 다양하게 학습
            durations = [0.5, 0.7]
            
            # ✅ 유지: swing_height는 좋은 범위
            swing_height = [0.04, 0.08]

        # ========== 개선안 B: 더 엄격한 제자리 걷기 (주석) ==========
        # 이 옵션은 아래 주석을 해제하고 위 A를 주석처리하면 됨
        # class ranges:
        #     frequencies = [0.9, 1.1]       # 좁은 주파수 범위
        #     offsets = [0.48, 0.52]         # 거의 완벽한 180도만
        #     durations = [0.55, 0.65]       # 좁은 stance/swing 범위
        #     swing_height = [0.04, 0.08]

    class init_state:
        pos = [0.0, 0.0, 0.9]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        
        default_joint_angles = {
            "torso_yaw_joint": 0.0,
            "R_hip_pitch_joint": -0.125,
            "R_hip_roll_joint": -0.03,
            "R_hip_yaw_joint": 0.0,
            "R_knee_pitch_joint": -0.25,
            "R_ankle_pitch_joint": 0.12,
            "R_ankle_roll_joint": 0.0,
            "L_hip_pitch_joint": -0.125,
            "L_hip_roll_joint": 0.03,
            "L_hip_yaw_joint": 0.0,
            "L_knee_pitch_joint": -0.25,
            "L_ankle_pitch_joint": -0.12,
            "L_ankle_roll_joint": 0.0,
        }

    class control:
        action_scale = 0.025
        control_type = "P"
        stiffness = {
            "torso_yaw_joint": 100.0,
            "R_hip_pitch_joint": 100.0,
            "R_hip_roll_joint": 100.0,
            "R_hip_yaw_joint": 60.0,
            "R_knee_pitch_joint": 150.0,
            "R_ankle_pitch_joint": 60.0,
            "R_ankle_roll_joint": 60.0,
            "L_hip_pitch_joint": 100.0,
            "L_hip_roll_joint": 100.0,
            "L_hip_yaw_joint": 60.0,
            "L_knee_pitch_joint": 150.0,
            "L_ankle_pitch_joint": 60.0,
            "L_ankle_roll_joint": 60.0,
        }
            
        damping = {
            "torso_yaw_joint": 5.0,
            "R_hip_pitch_joint": 5.0,
            "R_hip_roll_joint": 5.0,
            "R_hip_yaw_joint": 3.0,
            "R_knee_pitch_joint": 8.0,
            "R_ankle_pitch_joint": 4.0,
            "R_ankle_roll_joint": 4.0,
            "L_hip_pitch_joint": 5.0,
            "L_hip_roll_joint": 5.0,
            "L_hip_yaw_joint": 3.0,
            "L_knee_pitch_joint": 8.0,
            "L_ankle_pitch_joint": 4.0,
            "L_ankle_roll_joint": 4.0,
        }
            
        decimation = 4
        user_torque_limit = 120.0
        torque_limits = {
            "torso_yaw_joint": 60.0,
            "R_hip_pitch_joint": 60.0,
            "R_hip_roll_joint": 60.0,
            "R_hip_yaw_joint": 60.0,
            "R_knee_pitch_joint": 120.0,
            "R_ankle_pitch_joint": 17.0,
            "R_ankle_roll_joint": 17.0,
            "L_hip_pitch_joint": 60.0,
            "L_hip_roll_joint": 60.0,
            "L_hip_yaw_joint": 60.0,
            "L_knee_pitch_joint": 120.0,
            "L_ankle_pitch_joint": 17.0,
            "L_ankle_roll_joint": 17.0,
        }
        max_power = 1000.0

    class asset:
        terminate_after_contacts_on = ["base", "abad", "hip", "knee"]
        file = "/home/kim/kudos_ws/qub/src/QUB_urdf/urdf/qub_gpu.urdf"
        name = "qub"
        foot_name = "foot"
        foot_radius = 0.03
        penalize_contacts_on = ["knee", "hip"]
        
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False

        density = 0.001
        angular_damping = 1.0
        linear_damping = 1.0
        max_angular_velocity = 100.0
        max_linear_velocity = 1000.0
        armature = 0.05
        thickness = 0.01

    class domain_rand:
        randomize_friction = False
        friction_range = [0.0, 1.6]
        randomize_restitution = False
        restitution_range = [0.0, 1.0]
        randomize_base_mass = False
        added_mass_range = [-0.5, 5]
        randomize_base_com = False
        rand_com_vec = [0.03, 0.02, 0.03]
        randomize_inertia = False
        randomize_inertia_range = [0.8, 1.2]
        push_robots = False
        push_interval_s = 7
        max_push_vel_xy = 1.5
        rand_force = False
        force_resampling_time_s = 15
        max_force = 50.0
        rand_force_curriculum_level = 0
        randomize_Kp = False
        randomize_Kp_range = [0.8, 1.2]
        randomize_Kd = False
        randomize_Kd_range = [0.8, 1.2]
        randomize_motor_torque = False
        randomize_motor_torque_range = [0.8, 1.2]
        randomize_default_dof_pos = False
        randomize_default_dof_pos_range = [-0.05, 0.05]
        randomize_action_delay = False
        randomize_imu_offset = False
        randomize_imu_offset_range = [-1.2, 1.2]
        delay_ms_range = [0, 20]

    class rewards:
        class scales:
            # ========== 개선안 A: 자연스러운 제자리 걷기 ==========
            
            # 🔴 4차 개선(round 5c): 0.0 → 0.1
            # 설명: 이 코멘트 자체가 "0.0 → 0.1"이라고 예전부터 적혀있었는데 실제 코드값은
            # 계속 0.0이었음 — 의도했던 수정이 반영이 안 된 채로 방치돼있던 버그.
            # _reward_keep_balance는 매 스텝 그냥 +1(생존 보너스)이라 pointfoot 원본은
            # 1.0(=tracking_lin_vel 원본 스케일 1과 맞먹는 크기)으로 씀. QUB는 이미 다른
            # 스케일들이 원본보다 훨씬 큰 값들(tracking_lin_vel=8.0 등)이라 1.0은 상대적으로
            # 너무 작아 무의미할 수 있어 원래 문서화됐던 의도값 0.1로 되살림.
            keep_balance = 0.1

            # 🔴 2차 개선: 2.0/1.0 → 8.0/2.0
            # 설명: tracking_lin_vel을 2.0까지 올렸는데도 contact-shaped 계열이
            # 워낙 커서(100/40) "제자리에서 발만 리드미컬하게 움직이는" 정책에
            # 정체됨 (실측 actual_vx≈0.03 vs cmd_vx=0.5). 전진 추종 보상을 더 키움.
            tracking_lin_vel = 8.0
            tracking_ang_vel = 2.0

            # 🔴 3차 개선: 30.0/15.0 → -2.0/-2.0
            # 설명: 실제 원본(pointfoot_flat_config.py, 이 저장소 안에서 검증된 다른
            # 로봇 설정)을 직접 대조해보니 -2.0/-2.0(약한 페널티 분기)였음. 이 리워드
            # 함수는 스케일 부호에 따라 분기가 바뀌는 구조라 부호 자체는 버그가 아니지만,
            # 우리가 선택한 "강한 양의 보상" 분기(최대 걸음당 30+15=45)가 tracking_lin_vel
            # (8.0)보다 압도적으로 커서 "제자리에서 gait-clock만 맞추는" 정책이 계속
            # 최적이 됐던 것으로 보임. 원본과 같은 분기/스케일로 되돌려서 tracking_lin_vel이
            # 주도권을 갖게 함.
            tracking_contacts_shaped_force = -2.0
            tracking_contacts_shaped_vel = -2.0

            # 🔴 3차 개선: -4.0 → -50.0
            # 설명: "원본 -5.0에 가깝게 복원"이라던 이전 코멘트가 틀린 기억값이었음 —
            # 실제 pointfoot_flat_config.py 원본은 -100.0. QUB의 min_feet_distance(0.08)가
            # pointfoot(0.115)보다 좁아서 위반 시 자연스러운 벌어짐 정도도 더 작을 것으로
            # 보고 절반 정도(-50.0)로 비례 조정. -4.0으로는 옆으로 벌리는 것을 막기에
            # 턱없이 약했음(실측: 실제로 다리를 옆으로 벌리는 정책으로 수렴 확인).
            feet_distance = -50.0
            
            feet_regulation = -0.05
            
            # 🔴 원본: -0.1 → 🟢 개선: -0.5
            # 설명: 발이 부드럽게 착지하도록 강화
            foot_landing_vel = -0.5
            
            # 🔴 원본: -0.5 → 🟢 개선: -0.1
            # 설명: 기본(직립) 관절 각도에서 벗어나는 것에 대한 벌점이 커서
            # 보행에 필요한 다리 스윙 동작 자체를 억제하고 있었음. 완화함.
            dof_pos = -0.1
            base_height = -5.0
            # 🔴 4차 개선(round 5c): -10.0 → -2.0 (pointfoot 원본과 동일하게)
            # 설명: soft_dof_pos_limit(아래)과 겹쳐서 관절 가동범위를 이중으로 옥죄고
            # 있었음 — QUB는 원본보다 5배 센 벌점을, 게다가 훨씬 좁은 범위(75%)부터
            # 적용 중이었음. 진짜 보행에 필요한 큰 힙/무릎 굽힘 자체를 억제했을 가능성.
            dof_pos_limits = -2.0
            
            # 🔴 원본: -3.0 → 🟢 개선: -5.0
            # 설명: 로봇이 기울지 않도록 더 강하게
            orientation = -5.0
            
            # ✅ 나머지 페널티 (유지)
            lin_vel_z = -0.5
            ang_vel_xy = -0.05
            torques = -0.00005
            dof_acc = -1.0e-7
            action_rate = -0.005
            action_smooth = -0.03
            collision = -1.0
            feet_swing_height = -20.0

            # 신규 추가: feet_air_time (qub_flat.py에 새로 구현, 죽어있던 인프라를 살림)
            # 설명: tracking_lin_vel/contacts_shaped 같은 추상적인 결과 보상만으로는
            # "실제로 발을 뗀다"는 행동 자체에 대한 직접적인 유인이 부족해서 5000
            # iteration까지도 한 발짝 떼고 다시 정지하는 패턴이 반복됨. 착지 시점에만
            # 터지는 보상이라(스텝마다 X) 다른 항목 대비 스케일을 크게 잡음.
            # 🔴 3차 개선: 원래 방향 무관(스윙→착지만 보면 보상)이었던 게 8000 iteration
            # 실측/시각 확인 결과 "옆으로 다리 벌리기"로 farming되는 게 확인됨 — pointfoot/
            # wheelfoot 원본엔 애초에 이 리워드 자체가 없음(우리만의 커스텀 추가). qub_flat.py
            # 쪽에서 커맨드 방향 변위가 있을 때만 보상하도록 게이팅 로직을 추가했으므로
            # 스케일은 유지하고 로직 수정으로 대응.
            feet_air_time = 10.0

        # ========== 개선안 B: 더 엄격한 제자리 걷기 (주석) ==========
        # 아래를 주석 해제하고 위 A를 주석처리하면 더 엄격한 설정이 됨
        # class scales:
        #     keep_balance = 0.2             # 생존 보상 강화
        #     tracking_lin_vel = 0.5
        #     tracking_ang_vel = 0.2
        #     tracking_contacts_shaped_force = 100.0
        #     tracking_contacts_shaped_vel = 100.0
        #     feet_distance = -3.0           # 다리 모으기 강화
        #     feet_regulation = -0.05
        #     foot_landing_vel = -0.8        # 착지 품질 강화
        #     dof_pos = -3.0
        #     base_height = -5.0
        #     dof_pos_limits = -10.0
        #     orientation = -8.0             # 자세 제어 강화
        #     lin_vel_z = -0.5
        #     ang_vel_xy = -0.05
        #     torques = -0.00005
        #     dof_acc = -1.0e-7
        #     action_rate = -0.005
        #     action_smooth = -0.01
        #     collision = -1.0

        only_positive_rewards = False
        clip_reward = 100
        clip_single_reward = 5
        tracking_sigma = 0.2
        ang_tracking_sigma = 0.25
        height_tracking_sigma = 0.01
        # 🔴 4차 개선(round 5c): 0.75 → 0.95 (pointfoot 원본과 동일하게)
        # 설명: dof_pos_limits 스케일 완화(-10.0→-2.0)와 세트. 75%만 써도 벌점이 시작되던
        # 걸 원본처럼 95%까지 풀어줘서, 진짜 스텝에 필요한 큰 관절 가동범위를 정책이
        # 자유롭게 시도해볼 수 있게 함.
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.8
        base_height_target = 0.68
        feet_height_target = 0.10
        # 2차 개선: 0.12 -> 0.08 (원본 복원). feet_distance 완화(-1.0)랑 같이 작용해서
        # 다리를 넓게 벌린 정적 자세(기마자세)를 허용하던 원인이라 다시 좁힘.
        min_feet_distance = 0.08
        # 🔴 3차 신규 추가: max_feet_distance (원본 pointfoot엔 없지만 같은 저장소의
        # wheelfoot_flat_config.py엔 이미 있는 패턴). QUB 기본 직립 자세에서 실측한
        # 발 간격이 0.2665m(측정: 2026-08-28, headless reset 직후 5스텝)이라, 보행 중
        # 자연스러운 스탠스 폭 변동은 허용하되(+~30%) 8000 iteration에서 실제로 관찰된
        # "옆으로 다리 벌려 정적 안정성 버는" 행동은 확실히 막도록 0.35로 설정.
        max_feet_distance = 0.35
        about_landing_threshold = 0.06  # 🔴 원본: 0.08 → 🟢 개선: 0.06 (더 민감한 감지)
        max_contact_force = 100.0
        kappa_gait_probs = 0.05
        gait_force_sigma = 5.0
        gait_vel_sigma = 0.25
        gait_height_sigma = 0.005

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025
            height_measurements = 5.0
            contact_forces = 0.01
            torque = 0.05

        clip_observations = 100.0
        clip_actions = 100.0

    class noise:
        add_noise = True
        noise_level = 1.5

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class viewer:
        ref_env = 0
        pos = [5, -5, 3]
        lookat = [0, 0, 0]
        realtime_plot = True

    class sim:
        dt = 0.005
        substeps = 4
        gravity = [0.0, 0.0, -9.81]
        up_axis = 1

        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 16
            num_velocity_iterations = 1
            contact_offset = 0.02
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class QUBCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class MLP_Encoder:
        output_detach = True
        num_input_dim = QUBCfg.env.num_observations * QUBCfg.env.obs_history_length
        num_output_dim = 3
        hidden_dims = [256, 128]
        activation = "elu"
        orthogonal_init = False

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        orthogonal_init = False

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        # 🔴 4차 개선: 0.005 → 0.01 (pointfoot/wheelfoot 원본과 동일하게 복원)
        # 설명: iteration 6000에서 실측 결과 actual_vx std가 4000 대비 오히려 줄어들며
        # (0.7~1.1 → 0.02~0.5) 웅크린 채 정지하는 새 정적 국소최적해로 수렴 중인 게 확인됨.
        # feet_distance 상한/contacts_shaped 스케일 등 리워드 자체는 이미 원본에 맞춰
        # 고쳤는데도 같은 패턴이 재발한 걸로 봐서, 리워드 형태보다 탐색(exploration)
        # 부족이 근본 원인일 가능성이 커짐 — entropy_coef를 원본 수준으로 되돌려 정책이
        # 다양한 행동(특히 발을 떼는 시퀀스)을 더 오래 시도하도록 유도.
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

        est_learning_rate = 1.0e-3
        ts_learning_rate = 1.0e-4
        critic_take_latent = True

    class runner:
        encoder_class_name = "MLP_Encoder"
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 15000

        logger = "tensorboard"
        exptid = ""
        wandb_project = "legged_gym_PF"
        save_interval = 500
        experiment_name = robot_type
        run_name = ""
        resume = False
        load_run = "Aug27_14-28-19_"
        checkpoint = -1 
        resume_path = "None"