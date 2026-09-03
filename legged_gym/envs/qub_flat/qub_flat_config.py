# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# QUB humanoid - Phase 2 (제자리 보행)
#
# Phase 1 (v6): 안정적 standing 달성
# Phase 2: 잘 작동하는 v6 셋업 그대로 + gait 활성화 + 보행 reward 추가
#
# 변경 사항만 [P2] 주석. 나머지는 모두 Phase 1 그대로.
# Phase 1 체크포인트(May11_22-58-02_)에서 resume
#
# 2026-08-29: 친구가 준 이 파일(다운로드 폴더)을 기반으로 우리 저장소(kudos_ws)에
# 이식. 클래스명(QubCfg->QUBCfg 등)만 우리 envs/__init__.py의 등록 이름에 맞춰 변경,
# asset.file 경로를 우리 URDF 실경로로 교체, num_envs를 이 GPU(12GB) 기준으로 축소,
# runner.resume/load_run을 우리 쪽엔 없는 체크포인트라 resume=False로 변경. 그 외
# reward/gait/커맨드 등 실제 학습 레시피는 전부 원본 그대로 유지 (검증된 값이므로
# 임의로 건드리지 않음).

from legged_gym.envs.base.base_config import BaseConfig
from legged_gym import LEGGED_GYM_ROOT_DIR
import os


class QUBCfg(BaseConfig):
    class env:
        # 4096 -> 2048 -> 🔴 4096 복원 (2026-09-02).
        # 처음 2048로 낮춘 근거였던 "1024env=4.8GB"는 이전 config 기준 수치라 지금과
        # 맞지 않았음. 이 config로 실측하니 2048env에서 학습 프로세스 3.85GB /
        # GPU 12.3GB(RTX 4070 SUPER) 사용 = env당 약 1.15MB. 4096이면 약 6.2GB로
        # 충분히 들어감. 친구 원본값이기도 하고, 병렬 수가 많을수록 매 PPO 업데이트의
        # 경험이 다양해져 학습이 안정적이라 원본대로 복원.
        num_envs = 4096
        num_observations = 51
        num_critic_observations = 3 + num_observations
        num_height_samples = 117
        num_actions = 13
        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 20
        obs_history_length = 10
        dof_vel_use_pos_diff = False
        fail_to_terminal_time_s = 2.0

    class terrain:
        mesh_type = "plane"
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 1.5
        dynamic_friction = 1.5
        restitution = 0.0
        measure_heights = False
        critic_measure_heights = True
        measured_points_x = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 9
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
        heading_command = False
        min_norm = 0.1
        # 🔴 [P3] 2026-09-02: 0.0 -> 0.8. 제자리 박자가 4000 iter에서 확실히 잡혀서
        # (좌우 교대 스윙 4~4.8cm, base_h 0.85 고정) 이제 전진 커맨드를 여는 단계.
        # ⚠️ 이름과 달리 동작이 반대임: _resample_commands가 `rand > zero_command_prob`로
        # 판정해서 0.0이면 전부 0으로 덮어씀. 즉 이 값은 "커맨드를 받는 비율"에 가까움.
        # 0.8 = 80%는 명령을 받고, 20%는 계속 제자리(0) 유지 → 기껏 배운 제자리 박자를
        # 잃지 않으면서 커맨드 반응을 새로 배우게 함.
        zero_command_prob = 0.8

        class ranges:
            # [P2] Phase 2 초반: command 여전히 0. 제자리에서 박자만 학습.
            # 박자가 잡힌 후 (~2000 iter) 다음 단계에서 lin_vel_x = [-0.2, 0.2] 부터 열기
            # 🔴 [P3] 위 계획대로 전진부터 개방. 지금까지 세 축 모두 0만 겪어봤으므로
            # (커맨드가 전부 0으로 덮어써졌음) 한꺼번에 넓게 열지 않고 좁게 시작함.
            # x/y는 update_command_curriculum이 성공할 때마다 ±0.05씩 자동으로 넓혀줌.
            # 🔴🔴 [P3-b] 2026-09-02: 0.2 -> 0.5. **커리큘럼이 실제로는 동작하지 않음**을
            # 발견해서 직접 넓힘. 연결고리:
            #   _update_terrain_curriculum: move_up = (이동거리 > terrain_length/2 = 4.0m)
            #   → terrain_levels 상승 → max_terrain_level 도달해야 success_ids에 들어감
            #   update_command_curriculum: `if ... and len(self.success_ids) != 0` 에서 막힘
            # 에피소드 20초 × 실제속도 0.174m/s(명령 0.2) ≈ 3.5m < 4.0m 이라 승급 자체가
            # 불가능 → success_ids가 항상 비어서 범위 확대 코드가 한 번도 실행 안 됨.
            # 실제로 2000→5000 iteration 동안 한계가 0.6에서 전혀 안 올랐음(스윕 실측).
            # 그래서 커리큘럼에 기대지 않고 실측치 기반으로 직접 지정:
            #   실측 한계 0.6(생존100%/오차13.5%), 0.7에서 붕괴(생존17.7%).
            #   학습 중엔 resampling_time=5.0으로 명령이 급변하고(급출발 조건 0.6=생존74%)
            #   안전마진이 필요해서 0.5로 설정.
            # ⚠️ 커리큘럼이 죽어있으므로 앞으로도 speed_sweep.py로 재측정 → 수동으로
            #    이 값을 올리는 방식으로 진행해야 함.
            lin_vel_x = [-0.5, 0.5]
            lin_vel_y = [-0.1, 0.1]   # 측면은 실측한 적 없어서 그대로 둠 (전진에 집중)
            # ⚠️ ang_vel_yaw는 커리큘럼 확장 대상이 아님(base_task.py는 x/y만 넓힘).
            # 여기 적은 값이 끝까지 고정이므로, 회전이 필요해지면 나중에 직접 넓혀야 함.
            # 지금은 전진 학습이 목표라 ±1.0은 과해서 ±0.3으로 줄임.
            ang_vel_yaw = [-0.3, 0.3]
            heading = [-3.14, 3.14]

    class gait:
        num_gait_params = 4
        resampling_time = 5

        class ranges:
            # [P2] Gait 활성화 - 석사님 말씀대로 60% stance + 40% swing
            frequencies = [1.7, 1.7]      # 1.7Hz (사람 평지 걸음과 비슷)
            offsets = [0.5, 0.5]          # 좌우 다리 180도 위상차 (교대로)
            durations = [0.6, 0.6]        # 60% stance
            swing_height = [0.03, 0.03]   # swing 시 발 6cm 들기

    class init_state:
        pos = [0.0, 0.0, 0.95]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            "torso_yaw_joint": 0.0,
            "R_hip_pitch_joint": -0.1,
            "R_hip_roll_joint": -0.03,
            "R_hip_yaw_joint": 0.0,
            "R_knee_pitch_joint": -0.204,
            "R_ankle_pitch_joint": 0.1,
            "R_ankle_roll_joint": 0.0,
            "L_hip_pitch_joint": -0.1,
            "L_hip_roll_joint": 0.03,
            "L_hip_yaw_joint": 0.0,
            "L_knee_pitch_joint": -0.204,
            # 🔴 [P3-k] 2026-09-03 +0.1 -> -0.1. 기본자세 자체가 좌우 비대칭이던 버그.
            #   URDF에서 ankle_pitch만 좌우 축이 반대다(R (0,-1,0) / L (0,+1,0)).
            #   나머지 11개 다리 관절은 좌우 축이 같고, 12개 관절 origin이 전부
            #   rpy="0 0 0"이라 모든 링크 프레임이 베이스와 정렬돼 있다. 따라서 축이
            #   반대인 ankle_pitch만 '같은 값이 정반대 물리 회전'을 만든다.
            #   그런데 포팅해온 config는 L도 R도 +0.1이었다.
            #   실측(default_pose_check.py, 정책 없이 액션 0):
            #     foot0(L) pitch +11.99도 / foot1(R) pitch +1.44도 -> 차이 10.55도
            #   즉 로봇이 가만히 서 있는 기본자세부터 한 발은 발끝이 들리고 다른 발은
            #   내려가 있었다. posture(-0.2)와 ankle_regularization(-0.05)이 이 비대칭
            #   자세를 계속 목표로 삼으므로, joint_symmetry(-20)와 정면으로 싸운다.
            #   그 줄다리기 결과 L_ankle_pitch가 -17.49도까지 끌려갔고(발끝 들림 =
            #   뒤꿈치 박힘), 6개 관절 중 ankle_pitch만 유일하게 악화됐다
            #   (-2.96 -> -5.12도, 나머지 5개는 모두 크게 개선).
            #   -0.1이면 L+R=0이 되어 기본자세가 대칭이 되고, posture /
            #   ankle_regularization / joint_symmetry가 처음으로 같은 방향을 가리킨다.
            "L_ankle_pitch_joint": -0.1,
            "L_ankle_roll_joint": 0.0,
        }

    class control:
        # [P2] Phase 1 그대로 유지 - 잘 작동하는 셋업 건드리지 말기
        action_scale = 0.1
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
        # 🔴 친구 원본의 resources/robots/qub/urdf/QUB.urdf 경로가 이 저장소엔 없어서
        # 우리가 실제 쓰던 절대경로로 교체 (같은 URDF 파일, 배치만 다름 - 사용자 확인함)
        file = "/home/kim/kudos_ws/qub/src/QUB_urdf/urdf/qub_gpu.urdf"
        name = "qub"
        foot_name = "foot"
        foot_radius = 0.03
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base_link", "pelvis_link"]
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False
        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 100.0
        max_linear_velocity = 100.0
        armature = 0.05
        thickness = 0.01

    class domain_rand:
        # 1. 바닥 마찰력 랜덤화 (얼음판부터 끈적한 고무바닥까지)
        randomize_friction = False
        friction_range = [0.2, 1.25] 

        # 2. 로봇 몸무게 랜덤화 (배터리나 부품이 추가/변경되는 상황 대비, -1kg ~ +3kg)
        randomize_base_mass = False
        added_mass_range = [-1.0, 3.0]

        # 3. 무게 중심 위치 무작위 이동 (조립 오차 대비)
        randomize_base_com = False
        rand_com_vec = [0.03, 0.02, 0.03] # x, y, z 방향으로 최대 3cm 오차

        # 4. 외부 충격 (투명 인간이 10초마다 1.0m/s 속도로 로봇을 뻥뻥 걷어참)
        push_robots = False
        push_interval_s = 10
        max_push_vel_xy = 1.0

        # 5. 모터 제어 지연 (PC와 CAN 통신 사이의 랙(Lag) 시뮬레이션)
        randomize_action_delay = False
        delay_ms_range = [0, 20] # 0 ~ 20ms 사이의 랜덤한 통신 지연 발생

        # 6. 모터 토크 및 Kp, Kd 오차 (실제 모터 스펙이 카탈로그와 다를 때 대비)
        randomize_motor_torque = False
        randomize_motor_torque_range = [0.85, 1.15] # 힘이 15% 빠지거나 15% 쎄짐
        randomize_Kp = False
        randomize_Kp_range = [0.9, 1.1]
        randomize_Kd = False
        randomize_Kd_range = [0.9, 1.1]

        # 나머지 안 쓰는 것들은 False
        randomize_restitution = False
        restitution_range = [0.0, 0.2]
        randomize_imu_offset = False
        randomize_imu_offset_range = [-0.05, 0.05]
        randomize_inertia = False
        randomize_inertia_range = [0.8, 1.2]
        randomize_default_dof_pos = False
        randomize_default_dof_pos_range = [-0.05, 0.05]

    class rewards:
        class scales:
            # ============================================================
            # [P2] Phase 2: Standing reward 유지 + Gait reward 추가
            # 핵심: posture를 약화시켜야 다리 들 수 있음
            # ============================================================

            # --- alive (그대로) ---
            keep_balance = 2.0

            # --- 자세 (Phase 1 유지하되 posture만 약화) ---
            orientation = -1.0
            base_height = -3.0
            # [P2] -2.0 -> -0.5 : 다리를 들어야 보행이라 default pose에서 벗어남이 필요
            posture = -0.2
            ankle_regularization = -0.05   # [P2] -1.0 -> -0.5 약화

            # --- 주저앉기 차단 (그대로) ---
            feet_distance_max = -3.0

            # --- regularization (그대로) ---
            torques = -0.00005
            action_rate = -0.1
            dof_pos_limits = -1.0

            # --- [P2 신규] Gait reward (보행의 핵심) ---
            tracking_contacts_shaped_force = 3.0  # stance phase: 발 GRF 받기 (땅에 붙기)
            tracking_contacts_shaped_vel = 1.5    # swing phase: 발 속도 내기 (공중에 뜨기)
            foot_landing_vel = -0.5               # 부드러운 착지 (충격 줄임)
            feet_regulation = -0.05               # 발 컨트롤 (불필요한 움직임 억제)

            # --- 보행 후 활성화 ---
            tracking_lin_vel = 1.5    # command range 열 때 함께 활성화
            tracking_ang_vel = 1.5
            lin_vel_z = -2.0           # 보행 시 vz가 약간 있는 게 정상이라 일단 OFF
            ang_vel_xy = -0.05
            action_smooth = 0.0
            dof_acc = 0.0
            collision = 0.0
            feet_distance = 0.0
            ang_vel_z = 0.0
            yaw_drift = 0.0
            # ⚠️ symmetry는 계속 0으로 둔다. 같은 시점의 좌우 액션을 비교하는 구현이라
            # 180도 위상차인 보행을 벌하게 되고, 양발을 같이 움직이는 제자리뛰기 쪽으로
            # 밀려난다. 보폭 대칭은 아래 step_symmetry(위상 고려)로 처리한다.
            symmetry = 0.0

            # 🔴 [P3-c] 2026-09-02 신규: 좌우 보폭 비대칭 페널티.
            # 실측(model_5000, cmd 0.4): 좌우 발의 몸기준 평균 전후위치가 9.4cm 차이,
            # 한쪽 발은 몸 앞으로 아예 안 나오는 절뚝 보행이었음.
            # 스케일 산정: 현재 오차 0.094m -> 제곱 0.0088 -> ×2.0 = 0.0176/스텝
            #   ×50스텝/초 = 약 0.88/초. keep_balance(2.0/초)의 44% 수준이라 확실히
            #   체감되면서 다른 항목을 압도하진 않는 크기. 제곱이라 개선될수록 빠르게
            #   작아짐(5cm->0.25/초, 2cm->0.04/초)이므로 초반에 강하게 밀어준다.
            #
            # 🔴 [P3-d] 2026-09-02 -2.0 -> -5.0. 실측 경과: 9.4cm -> 6.9cm(500 iter)
            #   -> 5.72cm(Sep02_01-49-09_/model_2000, cmd 0.4, 64/64 생존).
            #   줄고는 있으나 속도가 급감(첫 500에 2.5cm, 다음 1500에 1.2cm).
            #   원인은 위 주석이 예고한 제곱항의 자기소멸이다. 오차가 줄면 페널티도
            #   제곱으로 줄어 9.4cm에서 0.88/초였던 압력이 5.72cm에서는 0.33/초
            #   (keep_balance의 16%)까지 떨어졌다. 즉 '충분히 밀었는데 안 되는' 것이
            #   아니라 '밀는 힘이 사라진' 상태.
            #   -5.0이면 5.72cm에서 0.0572^2 x 5.0 x 50 = 0.82/초 = keep_balance의 41%로,
            #   처음 설계했던 압력(0.88/초)으로 되돌리는 것이지 새로 과하게 조이는 게
            #   아니다. clip_single_reward=5와도 무관(스텝당 0.016 수준).
            #   남은 비대칭의 성질도 이 선택을 지지한다: 보폭은 좌우가 0.1481/0.1488로
            #   사실상 동일하고 차이는 순수한 DC 오프셋(한쪽 궤적 전체가 뒤로 밀림)이라,
            #   '평균 전후위치 차이'를 직접 벌하는 이 항이 정확히 맞는 표적이다.
            #
            # 🔴 [P3-e] 2026-09-02 -5.0 -> -2.0 (약화). 값을 내리는 게 아니라 역할을
            #   나누는 것이다. 아래 foot_landing_symmetry가 '착지 지점'이라는 더 정확한
            #   표적을 맡게 되는데, 둘은 사실상 같은 양을 재므로 둘 다 -5로 두면 이중
            #   계산이 된다. step_symmetry는 넓게 깔아주는 shaping 항으로 남긴다.
            #   또한 이 항의 기준 프레임이 base -> pelvis로 바뀌었으므로(qub_flat.py)
            #   같은 스케일이라도 실제로 보는 오차가 5.72cm -> 7.02cm로 커진다.
            # 🔴 [P3-i] 2026-09-02 -2.0 -> -30.0. 스케일 산정 자체가 틀렸던 것을 고친다.
            #   legged_gym 로그의 rew_xxx는 raw x scale의 시간평균이다
            #   (episode_sums += raw x scale x dt 를 max_episode_length_s로 나누므로
            #    dt와 스텝수가 상쇄된다). 그런데 이 파일의 이전 주석들은 여기에 50을
            #   또 곱해 "0.8/초 = keep_balance의 39%"처럼 계산했다. 50배 과대평가였다.
            #   실제로는 5.50cm에서 0.055^2 x 2.0 = 0.0061 = keep_balance(1.99)의 0.3%로
            #   사실상 놀고 있었다. 검산: 로그 -0.0027/2.0 -> 3.67cm vs 실측 3.69cm 일치.
            #   -30.0이면 0.091로 posture(-0.13)와 같은 급이 된다.
            step_symmetry = -30.0

            # 🔴 [P3-e] 2026-09-02 신규 3종. 실측 근거는 qub_flat.py의 __init__ 주석 참고.
            #
            # torso_yaw_drift: 골반을 한쪽으로 비틀어 고정하는 것(DC)만 벌한다.
            #   실측 DC +3.40도(=0.0593rad), 개체간 표준편차 0.01도로 64대가 전부 동일.
            #   보행 중 골반 스윙(AC +-1.9도)은 정상이라 살려야 하므로 EMA로 DC만 분리.
            #   스케일: 0.0593^2 x 5.0 x 50 = 0.88/초 = keep_balance(2.0/초)의 44%.
            #   step_symmetry를 처음 설계했을 때와 같은 압력 수준으로 맞춘 것.
            #   1도로 줄면 0.076/초, 0.5도면 0.019/초로 빠르게 사라진다.
            torso_yaw_drift = -5.0

            # foot_landing_symmetry: 좌우 발이 '딛는 위치'를 같게. 사용자가 지목한 표적.
            #   착지 이벤트에서만 갱신되는 EMA라 평균위치(step_symmetry)와 달리
            #   스윙 프로파일이 좌우로 달라도 잡아낸다.
            #   착지위치 차이는 아직 미측정이라 평균위치와 비슷한 7cm 가정.
            #   0.07^2 x 3.0 x 50 = 0.74/초. step_symmetry(-2.0, 0.49/초)와 합쳐
            #   보폭 대칭 압력이 1.2/초 수준이 된다.
            # 🔴 [P3-i] -3.0 -> -30.0. step_symmetry와 같은 50배 계산 오류 정정.
            foot_landing_symmetry = -30.0

            # stride_symmetry: 좌우 보폭(이지->착지 전진거리)을 같게.
            #   ⚠️ 실측상 보폭은 이미 거의 같다(0.1481 vs 0.1488). 즉 지금 뭔가를 고치는
            #   항이 아니라, 위 항들을 강하게 누르는 과정에서 '평균은 맞췄는데 한쪽
            #   보폭이 줄어드는' 퇴행을 막는 보험이다. 계속 0 근처면 정상.
            #
            # 🔴 [P3-f] 2026-09-02 -2.0 -> -30.0. 위에서 우려한 퇴행이 실제로 일어났고,
            #   -2.0으로는 못 막았다. model_1000 실측: 보폭 L 0.1544 / R 0.1431로
            #   7.9% 차이(model_2000에서는 0.5%였다). 평균 위치 비대칭은 7.02->2.50cm로
            #   잘 줄었지만 그 과정에서 보폭 대칭이 무너진 것.
            #   -2.0이 무력했던 이유는 스케일 산정 착오다: 오차 0.0113m는 제곱하면
            #   1.3e-4라 -2.0에서 압력이 0.013/초, keep_balance의 0.6%에 불과했다.
            #   -30.0이면 0.19/초로 실제로 작동하는 크기가 된다. 제곱항이라 0.5cm까지
            #   줄면 0.038/초로 알아서 사라진다.
            stride_symmetry = -30.0

            # 🔴 [P3-f] 2026-09-02 신규. 좌우 hip_yaw 대칭.
            #   사용자가 뷰어에서 왼다리가 더 바깥으로 돌아간다고 관찰 -> 실측 확인:
            #     L_hip_yaw DC +22.13도 / R_hip_yaw DC -13.59도 -> 미러 잔차 +8.53도
            #   축이 좌우 동일(0,0,1)하므로 대칭 조건은 L+R=0. '차이'가 아니라 '합'을
            #   벌한다. 위상 문제 때문에 순간값이 아니라 EMA를 쓴다(qub_flat.py 주석 참고).
            #   스케일: 잔차 8.53도 = 0.1489rad -> 0.1489^2 x 0.7 x 50 = 0.78/초
            #   = keep_balance(2.0/초)의 39%. torso_yaw_drift와 같은 설계점.
            #   4도로 줄면 0.17/초, 2도면 0.043/초.
            #   ⚠️ 이 항은 좌우 균형만 잡는다. 양발이 똑같이 벌어진 대칭 팔자걸음
            #   (실측상 공통 성분 약 18도)은 L+R=0을 만족해 이 항으로는 안 줄어든다.
            #   -> 2026-09-02 사용자 판단: "발 약간 벌려서 걷는거 괜찮음". 공통 팔자는
            #      의도적으로 그대로 둔다. 대칭만 맞춘다.
            #
            # 🔴 [P3-g] -0.7 -> -2.0. model_2100 재측정에서 hip_yaw 잔차가 8.53 -> 3.23도로
            #   이미 저절로 줄어 있었다(hip_yaw 항 없이, 기존 발 대칭 항들의 효과).
            #   그래서 -0.7에서는 압력이 0.0564^2 x 0.7 x 50 = 0.11/초(keep_balance의 5.6%)
            #   밖에 안 됐다. 학습 로그에서 이 항이 예상보다 낮게(6%) 찍힌 것도 스케일
            #   오류가 아니라 이 때문이었다. 사용자가 명시적으로 요청한 항목이므로
            #   -2.0으로 올려 0.32/초(16%)로 맞춘다.
            # 🔴 [P3-i] 2026-09-02 -2.0 -> 0.0. 아래 joint_symmetry가 6관절을 한꺼번에
            #   다루므로 개별 항은 이중 계산이 된다. 개별 항 방식 자체가 실패였다:
            #   hip_roll만 집중해서 잡는 동안(6.58->0.88도) 앞뒤 발 위치가
            #   3.69->5.50cm로 악화됐다. 사용자 관찰과 일치("다른게 무너졌어").
            hip_yaw_symmetry = 0.0

            # 🔴 [P3-g] 2026-09-02 신규. 좌우 hip_roll 대칭 — 절뚝임의 실제 원인.
            #   model_2100 실측(cmd 0.4, 64/64 생존):
            #     L_hip_roll DC +6.98도 (진폭 1.87, 범위 +3.59~+10.23)
            #     R_hip_roll DC -0.01도 (진폭 0.05, 범위 -0.75~+0.02)  <- 사실상 고정
            #     미러 잔차 +6.97도
            #   R이 URDF 상한(0)에 붙어 얼어 있고 L만 7도 벌어진다. hip_yaw 잔차(3.23도)
            #   보다 두 배 크고, 발 heading 잔차 +4.89도의 주된 몫이다.
            #   원인은 정책이 아니라 soft limit 계산이다: R은 soft 범위를 만족하려면
            #   -6.9도보다 바깥으로 가야 하는데 그러지 않고 0에 머물며 dof_pos_limits를
            #   시간의 100% 동안 그냥 내고 있다(L은 48%).
            #   ★ 이 항과 dof_pos_limits는 충돌하지 않는다: L=+6.9 / R=-6.9 (대칭적 팔자)면
            #     둘 다 만족한다. 사용자가 팔자를 허용했으므로 이 해가 유효하다.
            #   스케일: 0.1216^2 x 1.0 x 50 = 0.74/초 = keep_balance의 37%.
            #   hip_yaw/torso_yaw와 같은 설계점. 3도로 줄면 0.14/초.
            hip_roll_symmetry = 0.0   # [P3-i] joint_symmetry로 통합. 위 주석 참고.

            # 🔴 [P3-i] 2026-09-02 신규. 다리 6관절 전체 미러 대칭 (개별 항들을 대체).
            #   model_1000(Sep02_23-22-07_) 실측 잔차와 비중:
            #     hip_pitch   -7.17도  0.01565 rad^2   59%   <- 한 번도 안 재봤던 관절
            #     knee_pitch  -3.26도  0.00324         12%
            #     ankle_roll  -2.93도  0.00260         10%
            #     ankle_pitch -2.96도  0.00268         10%
            #     hip_yaw     +2.76도  0.00232          9%
            #     hip_roll    +0.88도  0.00023        0.9%   <- 그동안 매달린 관절
            #     합 0.02672 rad^2
            #   즉 개별 항으로 잡던 hip_roll은 전체의 1%도 안 되고, 손대지 않던
            #   hip_pitch가 절반 이상이었다. 6쌍을 한 항으로 묶어야 하는 이유다.
            #   스케일 -20.0 -> 0.02672 x 20 = 0.534. keep_balance(1.99)의 27%,
            #   tracking_lin_vel(1.31)까지 합친 양의 보상 대비 16% 수준이라 보행을
            #   망가뜨리진 않으면서 확실히 지배적이다. 방침이 "무조건 대칭성"이고,
            #   이 크기여야 위 A(한쪽만 벌림) 선택지를 확실히 이긴다.
            #   제곱합이라 대칭이 잡힐수록 빠르게 사라진다(잔차 절반이면 1/4).
            joint_symmetry = -20.0

            # 🔴 [P3-l] 2026-09-03 신규. 좌우 관절의 '움직이는 크기'(진폭) 대칭.
            #   joint_symmetry의 구멍을 메운다: 그 항은 시간평균(DC)만 비교하므로
            #   한쪽 관절이 거의 굳어 있어도 평균만 미러면 잔차가 0으로 읽힌다.
            #   실측(Sep03_17-37-52_/model_1000)이 정확히 그 상태였다:
            #     ankle_pitch  L: DC -17.53 AC 3.39 / R: DC +15.45 AC 6.98
            #     -> DC 잔차 -2.08도(대칭)인데 진폭은 2배 차이.
            #   사용자가 화면에서 "왼발 ankle_pitch가 고정된 것 같고 뒤꿈치로 디딘다"고
            #   관찰한 것이 이것이다. 발목을 못 굽히니 뒤꿈치->발끝으로 구르지 못한다.
            #   실측 좌우 AC 차이(도): ankle_pitch 3.59 / ankle_roll 1.77 /
            #     knee_pitch 0.96 / hip_yaw 0.86 / hip_roll 0.46 / hip_pitch 0.24
            #   |편차| EMA 기준으로 환산한 제곱합 약 0.0044 rad^2.
            #   스케일 -20이면 0.089로 joint_symmetry(0.0736)와 대등하고,
            #   그 중 72%가 ankle_pitch 몫이라 문제 관절에 자동으로 집중된다.
            #   개별 관절 항을 또 만들지 않는 이유는 joint_symmetry와 같다 —
            #   제곱합이 알아서 가장 나쁜 관절로 압력을 옮긴다.
            #
            # ★ 2026-09-03 사용자 지시로 '왼쪽만 올리는' 단방향 항으로 바꿨다.
            #   (L-R)^2이면 정책이 '오른쪽을 줄여서' 대칭을 맞출 수 있는데, 오른발은
            #   이미 잘 걷고 있어 그러면 손해다("오른발은 건드리지 말자").
            #   그래서 식에서 R을 빼고 아래 amp_target_left(상수)에 미달할 때만 벌한다.
            #   현재 deficit 제곱합 약 0.0034 rad^2 -> x20 = 0.067로
            #   joint_symmetry(0.0736)와 대등하다.
            joint_amplitude_symmetry = -20.0

            # 🔴 [P3-m] 2026-09-03 신규. 왼발 접지 자세 교정 — 이번 문제의 정면 표적.
            #   실측(model_5000): 접지 순간 발 피치 L -7.88도 / R +1.19도, 차이 9.07도.
            #   스탠스 중에도 L -7.51도로 유지 -> 뒤꿈치로 닿아 그대로 있는다.
            #   ⚠️ 진폭 가설(joint_amplitude_symmetry)은 틀렸던 것으로 판명됐다:
            #      왼발 피치 '범위'는 18.2도로 오른발(6.2도)의 3배였다. 발목 관절
            #      진폭만 작았을 뿐(3.69 vs 6.48도) 발 피치는 다리 전체 자세에서
            #      나온다. 그래서 관절이 아니라 발을, 접지 순간에 겨냥한다.
            #   스케일: 편차 0.158rad -> 제곱 0.0251, 접지 구간 비율 약 0.6을 곱하면
            #   평균 0.015. x10 = 0.15로 action_rate(0.28)와 겨룰 수 있는 크기다.
            #   목표에 도달하면 0이 되어 스스로 꺼진다.
            # 🔴 [P3-n] 2026-09-03 -10.0 -> 0.0. 틀린 양을 겨냥하고 있었다.
            #   이 항은 '골반 기준 pitch'만 봤는데, 실측 결과 그 값이 -1.32도로
            #   평평해 보이는 동안에도 '땅 기준' 발바닥 기울기는 9.71도였다.
            #   원인 둘: (1) roll 미고려 - 기울기의 절반이 좌우 성분(-6.34도)이었다.
            #   (2) 기준 프레임 - 보행 중 골반이 앞으로 기울어 골반 기준과 땅 기준이
            #   7도 가까이 벌어진다. 아래 left_foot_flat_contact로 대체한다.
            left_foot_contact_pitch = 0.0

            # 🔴 [P3-n] 2026-09-03 신규. 왼발 발바닥이 땅에 전부 닿게 한다.
            #   사용자 관찰: "발 뒤꿈치 찍히는 정도는 아닌데 발바닥이 전부 땅에 닿지 않아"
            #   실측(Sep03_19-41-38_/model_3500, 발 좌표계에서 본 중력, 스탠스 중):
            #     L  x -7.00도  y -6.34도  -> 전체 기울기 9.71도   접촉시간 51.3%
            #     R  x -0.13도  y +1.05도  -> 전체 기울기 1.11도   접촉시간 56.4%
            #   오른발이 1.11도를 달성하고 있으므로 0은 도달 가능한 목표다.
            #   스케일: proj_g의 x^2+y^2 = 0.0271, 접촉 비율 0.513을 곱해 평균 0.0139.
            #   x15 = 0.21로 action_rate(0.28)와 겨룰 수 있는 크기다.
            #   목표가 0(절대값)이라 오른발을 나쁘게 만들어 만족시킬 길이 없다.
            left_foot_flat_contact = -15.0

            # 🔴 [P3-o] 2026-09-03 신규. 오른발도 발바닥 전체가 땅에 닿게 한다.
            #   사용자 관찰: "오른발 발 안쪽으로 걷는구나 왼발처럼 발바닥 다 쓰도록"
            #   실측(Sep03_20-50-02_/model_1500, 스탠스 중):
            #     R  x -0.06도  y +3.23도  -> 전체 3.24도
            #   기울기가 거의 전부 좌우(y) 성분이다. 앞뒤로는 평평한데 옆으로 돌아가
            #   안쪽 모서리로 딛고 있다는 뜻이고, 관찰과 정확히 일치한다.
            #   왼발 항을 넣기 전 오른발은 0.87도였는데 왼발이 9.93->0.22도로 잡히는
            #   동안 3.24도로 밀려났다. 오른발은 어떤 수식에도 없었으므로 직접적인
            #   압력이 아니라 보행 재최적화의 부수 효과로 보이고, 이제 명시적으로 잡는다.
            #
            #   스케일을 왼발(-15)보다 크게 잡는 이유: 제곱항이라 오차가 작을수록
            #   압력이 제곱으로 작아진다. 오른발 오차(3.24도)는 왼발 시작점(9.93도)의
            #   1/3이라 같은 스케일이면 압력이 1/8.5로 떨어져 사실상 놀게 된다.
            #   proj_g의 x^2+y^2 = 0.00318, 접촉 비율 0.513 -> 평균 0.00163.
            #   x50 = 0.082로 posture(0.13)와 같은 급이 된다. 0.6도까지 줄면 0.003.
            #
            #   ★ 좌우 두 항 모두 목표가 '상대값이 아니라 절대 0'이다. 그래서 둘을
            #   동시에 켜도 한쪽을 나쁘게 만들어 자기 항을 만족시키는 경로가 없다.
            right_foot_flat_contact = -50.0

        only_positive_rewards = False
        clip_reward = 100
        clip_single_reward = 5
        tracking_sigma = 0.2
        ang_tracking_sigma = 0.25
        height_tracking_sigma = 0.01
        soft_dof_pos_limit = 0.9

        # [P3-l] 2026-09-03. joint_amplitude_symmetry의 '왼쪽 진폭 목표'(rad).
        # 순서는 [hip_pitch, hip_roll, hip_yaw, knee_pitch, ankle_pitch, ankle_roll].
        # 값은 오른쪽 관절의 실측 진폭에서 뽑았다(Sep03_17-37-52_/model_1000, cmd 0.4).
        # 측정은 표준편차라 |편차| 평균으로 환산(정현파 기준 약 0.9배)했다:
        #   R AC(도) -> 목표(rad):  hip_yaw 3.08 -> 0.0484,  ankle_pitch 6.98 -> 0.1097
        #
        # 0으로 둔 관절은 '제약 없음'이다. 실측상 왼쪽이 부족한 건 두 관절뿐이고
        # (ankle_pitch -3.59도, hip_yaw -0.86도), 나머지 4개는 오히려 왼쪽이 더 크다
        # (hip_pitch +0.24 / hip_roll +0.46 / knee_pitch +0.96 / ankle_roll +1.77).
        # 잘 움직이는 관절에 괜히 목표를 걸면 깎아내릴 위험만 있으므로 건드리지 않는다.
        #
        # ⚠️ 상수이므로 보행 양상이 크게 바뀌면 다시 재서 갱신해야 한다. 반대로 상수인
        # 덕분에 정책이 '오른쪽을 줄여서' 이 항을 만족시키는 길이 원천 봉쇄된다.
        amp_target_left = [0.0, 0.0, 0.0484, 0.0, 0.1097, 0.0]

        # [P3-m] 2026-09-03. 왼발이 접지 중 유지해야 할 발 피치(rad, 골반 기준).
        # 오른발 실측치에서 뽑았다(Sep03_18-12-02_/model_5000):
        #   접지 순간  L -7.88도 / R +1.19도     스탠스 중  L -7.51도 / R +2.10도
        # 오른발은 거의 평평하게 닿는데 왼발만 9도 젖혀진 채 닿고 그대로 유지한다.
        # +1.19도 = 0.0208rad. 오른발 값을 상수로 박아두므로, 정책이 오른발을
        # 나쁘게 만들어 이 항을 만족시킬 방법이 없다.
        left_foot_contact_pitch_target = 0.0208
        # 🔴 [P3-j] 2026-09-02 False로 되돌림. 아래 보정을 끄고 원래 동작을 살린다.
        #   원래 동작은 dof_pos_limits가 양 hip_roll을 6.9도 이상 '벌리라'고 요구하는
        #   것이고, 이전에는 그것이 feet_distance_max와 충돌해 '한쪽만 벌리는' 절뚝임을
        #   만들었다. 그때 대칭 페널티가 없었기 때문이다. joint_symmetry(-20)를 넣으면
        #   비용 구조가 뒤집힌다 (rad 단위, 로그에 찍히는 크기 기준):
        #     A 한쪽만 벌림 : dof_pos_limits 0.12 + joint_symmetry 0.0144x20=0.288 = 0.408
        #     B 양쪽 대칭   : feet_distance_max 0.15                              = 0.150  <- 최선
        #     C 양쪽 안 벌림: dof_pos_limits 0.24                                 = 0.240
        #   즉 '벌리되 대칭으로'가 정책에게 가장 싼 선택이 된다. 관절 사용을 억제하는
        #   대신 대칭만 강제한다는 방침(사용자: "덜 쓰자가 아니라 대칭으로 쓰자")과 일치.
        #   B가 남기는 feet_distance_max 0.15가 거슬리면 max_feet_distance를 0.42로
        #   올리면 되지만, 스탠스 폭이 실제로 넓어지므로 먼저 결과를 보고 판단할 것.
        fix_soft_limit_to_include_default = False
        # [P3-h] 2026-09-02. soft limit이 기본자세를 배제할 때 경계를 물릴 여유(rad).
        # base_task의 중점 기준 공식이 QUB hip_roll처럼 범위가 치우친 관절에서
        # 기본자세조차 위반으로 만드는 문제를 막는다. 자세한 근거는 qub_flat.py의
        # _process_dof_props 주석 참고. 0.05rad(2.9도)는 기본자세 주변에서 관절이
        # 자유롭게 움직일 최소한의 여유이고, 하드 리밋을 넘지 않도록 클램프된다.
        soft_limit_default_margin = 0.05
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.8
        base_height_target = 0.70
        feet_height_target = 0.10
        min_feet_distance = 0.20
        max_feet_distance = 0.35
        # alive gate
        alive_min_height = 0.55
        alive_max_tilt = 0.85
        max_contact_force = 100.0
        kappa_gait_probs = 0.07
        gait_force_sigma = 50.0
        gait_vel_sigma = 0.5
        gait_height_sigma = 0.02
        about_landing_threshold = 0.08

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            torque = 0.05

        clip_observations = 30.0
        clip_actions = 1.0

    class noise:
        add_noise = True
        noise_level = 0.1

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.0
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05

    class viewer:
        ref_env = 0
        pos = [5, -5, 3]
        lookat = [0, 0, 0]
        realtime_plot = True

    class sim:
        dt = 0.005
        substeps = 2
        gravity = [0.0, 0.0, -9.81]
        up_axis = 1

        class physx:
            num_threads = 10
            solver_type = 1
            # 잘 작동하는 셋업 그대로
            num_position_iterations = 8
            num_velocity_iterations = 2
            contact_offset = 0.01
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
        init_noise_std = 0.8
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        orthogonal_init = False

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        critic_take_latent = True
        est_learning_rate = 1.0e-3
        ts_learning_rate = 1.0e-4

    class runner:
        encoder_class_name = "MLP_Encoder"
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 15000
        logger = "tensorboard"
        exptid = ""
        wandb_project = "legged_gym_QUB"
        save_interval = 100
        experiment_name = "qub_flat"
        run_name = ""
        resume = False
        load_run = "None"
        checkpoint = -1
        resume_path = "None"
