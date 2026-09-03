# QUB 보행 학습 최종 결과

최종 정책: `Sep04_00-19-14_/model_15000`

측정 조건은 `POLICY_PROGRESS.md` 와 동일. 경과별 추이는 그쪽 참고.

생성 시각: 2026-09-04 04:04:50

## foot_flatness
```
======================================================================
[QUBFlat] Robot index verification
----------------------------------------------------------------------
DOF order:
  [ 0] torso_yaw_joint
  [ 1] L_hip_pitch_joint
  [ 2] L_hip_roll_joint
  [ 3] L_hip_yaw_joint
  [ 4] L_knee_pitch_joint
  [ 5] L_ankle_pitch_joint
  [ 6] L_ankle_roll_joint
  [ 7] R_hip_pitch_joint
  [ 8] R_hip_roll_joint
  [ 9] R_hip_yaw_joint
  [10] R_knee_pitch_joint
  [11] R_ankle_pitch_joint
  [12] R_ankle_roll_joint
----------------------------------------------------------------------
feet_indices              : [7, 13]
penalised_contact_indices : [4, 10, 5, 11]
termination_contact_indices: [0, 1]
ankle DOF indices (auto)  : [5, 6, 11, 12]
======================================================================
'train_cfg' provided -> Ignoring 'name=qub_flat'
Encoder MLP: Sequential(
  (0): Linear(in_features=510, out_features=256, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=256, out_features=128, bias=True)
  (3): ELU(alpha=1.0)
  (4): Linear(in_features=128, out_features=3, bias=True)
)
Actor MLP: Sequential(
  (0): Linear(in_features=57, out_features=512, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=512, out_features=256, bias=True)
  (3): ELU(alpha=1.0)
  (4): Linear(in_features=256, out_features=128, bias=True)
  (5): ELU(alpha=1.0)
  (6): Linear(in_features=128, out_features=13, bias=True)
)
Critic MLP: Sequential(
  (0): Linear(in_features=60, out_features=512, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=512, out_features=256, bias=True)
  (3): ELU(alpha=1.0)
  (4): Linear(in_features=256, out_features=128, bias=True)
  (5): ELU(alpha=1.0)
  (6): Linear(in_features=128, out_features=1, bias=True)
)
Loading model from: /home/kim/kudos_ws/rl_library/tron1-rl-isaacgym/logs/qub_flat/qub_flat/Sep04_00-19-14_/model_15000.pt
======================================================================
checkpoint : Sep04_00-19-14_/model_15000   cmd_vx=0.4
foot0=L_foot_link, foot1=R_foot_link   생존 64/64
발 좌표계에서 본 중력. 발바닥이 땅과 평행하면 x=y=0.
  x = 앞뒤 기울기   y = 좌우 기울기(모서리로 딛는 정도)   tilt = 전체 각도
----------------------------------------------------------------------
[foot0 L_foot_link   ] 접지 순간: x  -0.40 deg  y  -0.44 deg  tilt  0.61 deg (sd 0.70)
[foot0 L_foot_link   ] 스탠스 중: x  -0.03 deg  y  -0.34 deg  tilt  0.36 deg (sd 0.34)
   접촉 시간 비율 : 51.1%
----------------------------------------------------------------------
[foot1 R_foot_link   ] 접지 순간: x  -0.15 deg  y  +0.00 deg  tilt  0.17 deg (sd 0.29)
[foot1 R_foot_link   ] 스탠스 중: x  -0.01 deg  y  +0.01 deg  tilt  0.02 deg (sd 0.09)
   접촉 시간 비율 : 51.1%
----------------------------------------------------------------------
해석: tilt가 0에 가까울수록 발바닥 전체가 땅에 닿는다.
======================================================================
```

## heel_clearance
```
======================================================================
[QUBFlat] Robot index verification
----------------------------------------------------------------------
DOF order:
  [ 0] torso_yaw_joint
  [ 1] L_hip_pitch_joint
  [ 2] L_hip_roll_joint
  [ 3] L_hip_yaw_joint
  [ 4] L_knee_pitch_joint
  [ 5] L_ankle_pitch_joint
  [ 6] L_ankle_roll_joint
  [ 7] R_hip_pitch_joint
  [ 8] R_hip_roll_joint
  [ 9] R_hip_yaw_joint
  [10] R_knee_pitch_joint
  [11] R_ankle_pitch_joint
  [12] R_ankle_roll_joint
----------------------------------------------------------------------
feet_indices              : [7, 13]
penalised_contact_indices : [4, 10, 5, 11]
termination_contact_indices: [0, 1]
ankle DOF indices (auto)  : [5, 6, 11, 12]
======================================================================
'train_cfg' provided -> Ignoring 'name=qub_flat'
Encoder MLP: Sequential(
  (0): Linear(in_features=510, out_features=256, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=256, out_features=128, bias=True)
  (3): ELU(alpha=1.0)
  (4): Linear(in_features=128, out_features=3, bias=True)
)
Actor MLP: Sequential(
  (0): Linear(in_features=57, out_features=512, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=512, out_features=256, bias=True)
  (3): ELU(alpha=1.0)
  (4): Linear(in_features=256, out_features=128, bias=True)
  (5): ELU(alpha=1.0)
  (6): Linear(in_features=128, out_features=13, bias=True)
)
Critic MLP: Sequential(
  (0): Linear(in_features=60, out_features=512, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=512, out_features=256, bias=True)
  (3): ELU(alpha=1.0)
  (4): Linear(in_features=256, out_features=128, bias=True)
  (5): ELU(alpha=1.0)
  (6): Linear(in_features=128, out_features=1, bias=True)
)
Loading model from: /home/kim/kudos_ws/rl_library/tron1-rl-isaacgym/logs/qub_flat/qub_flat/Sep04_00-19-14_/model_15000.pt
====================================================================
checkpoint : Sep04_00-19-14_/model_15000   cmd_vx=0.4
생존 64/64   발 박스 앞뒤 -0.090~+0.110 / 폭 0.090 m
--------------------------------------------------------------------
발 중심 거리        평균  20.81 cm   최소  19.67 cm
뒤꿈치 안쪽 모서리   평균   8.93 cm   최소   5.02 cm
두 발 최소 간격      평균   8.93 cm   최소   5.02 cm   <- 충돌 여유
--------------------------------------------------------------------
여유 5.02 cm - 충돌 위험 없음
====================================================================
```

## joint_sym_check
```
======================================================================
[QUBFlat] Robot index verification
----------------------------------------------------------------------
DOF order:
  [ 0] torso_yaw_joint
  [ 1] L_hip_pitch_joint
  [ 2] L_hip_roll_joint
  [ 3] L_hip_yaw_joint
  [ 4] L_knee_pitch_joint
  [ 5] L_ankle_pitch_joint
  [ 6] L_ankle_roll_joint
  [ 7] R_hip_pitch_joint
  [ 8] R_hip_roll_joint
  [ 9] R_hip_yaw_joint
  [10] R_knee_pitch_joint
  [11] R_ankle_pitch_joint
  [12] R_ankle_roll_joint
----------------------------------------------------------------------
feet_indices              : [7, 13]
penalised_contact_indices : [4, 10, 5, 11]
termination_contact_indices: [0, 1]
ankle DOF indices (auto)  : [5, 6, 11, 12]
======================================================================
'train_cfg' provided -> Ignoring 'name=qub_flat'
Encoder MLP: Sequential(
  (0): Linear(in_features=510, out_features=256, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=256, out_features=128, bias=True)
  (3): ELU(alpha=1.0)
  (4): Linear(in_features=128, out_features=3, bias=True)
)
Actor MLP: Sequential(
  (0): Linear(in_features=57, out_features=512, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=512, out_features=256, bias=True)
  (3): ELU(alpha=1.0)
  (4): Linear(in_features=256, out_features=128, bias=True)
  (5): ELU(alpha=1.0)
  (6): Linear(in_features=128, out_features=13, bias=True)
)
Critic MLP: Sequential(
  (0): Linear(in_features=60, out_features=512, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=512, out_features=256, bias=True)
  (3): ELU(alpha=1.0)
  (4): Linear(in_features=256, out_features=128, bias=True)
  (5): ELU(alpha=1.0)
  (6): Linear(in_features=128, out_features=1, bias=True)
)
Loading model from: /home/kim/kudos_ws/rl_library/tron1-rl-isaacgym/logs/qub_flat/qub_flat/Sep04_00-19-14_/model_15000.pt
========================================================================
checkpoint : Sep04_00-19-14_/model_15000   cmd_vx=0.4
alive      : 64/64   foot0=L_foot_link, foot1=R_foot_link
------------------------------------------------------------------------
[torso_yaw]  DC  +0.05 deg   AC  0.67 deg
------------------------------------------------------------------------
[hip_pitch]  mask +1  (대칭 조건 L = R)
  L: DC  -14.61  AC  5.51   R: DC  -14.95  AC  5.52
  >> 미러 잔차   +0.34 deg    제곱 0.00004 rad^2
------------------------------------------------------------------------
[hip_roll]  mask -1  (대칭 조건 L = - R)
  L: DC   +3.52  AC  3.04   R: DC   -3.44  AC  2.99
  >> 미러 잔차   +0.08 deg    제곱 0.00000 rad^2
------------------------------------------------------------------------
[hip_yaw]  mask -1  (대칭 조건 L = - R)
  L: DC  +19.64  AC  2.16   R: DC  -19.67  AC  2.18
  >> 미러 잔차   -0.03 deg    제곱 0.00000 rad^2
------------------------------------------------------------------------
[knee_pitch]  mask +1  (대칭 조건 L = R)
  L: DC  -32.85  AC 11.38   R: DC  -33.33  AC 11.56
  >> 미러 잔차   +0.48 deg    제곱 0.00007 rad^2
------------------------------------------------------------------------
[ankle_pitch]  mask -1  (대칭 조건 L = - R)
  L: DC  -15.17  AC  7.30   R: DC  +15.38  AC  7.53
  >> 미러 잔차   +0.21 deg    제곱 0.00001 rad^2
------------------------------------------------------------------------
[ankle_roll]  mask -1  (대칭 조건 L = - R)
  L: DC   +3.06  AC  2.14   R: DC   -3.16  AC  1.94
  >> 미러 잔차   -0.10 deg    제곱 0.00000 rad^2
------------------------------------------------------------------------
[6관절 제곱합] 0.00012 rad^2   (= _reward_joint_symmetry의 raw. scale -20이면 로그에 0.0025)
[발 heading]  foot0 +19.76  foot1 -19.76  잔차 +0.00 deg
========================================================================
```

## step_asym
```
======================================================================
[QUBFlat] Robot index verification
----------------------------------------------------------------------
DOF order:
  [ 0] torso_yaw_joint
  [ 1] L_hip_pitch_joint
  [ 2] L_hip_roll_joint
  [ 3] L_hip_yaw_joint
  [ 4] L_knee_pitch_joint
  [ 5] L_ankle_pitch_joint
  [ 6] L_ankle_roll_joint
  [ 7] R_hip_pitch_joint
  [ 8] R_hip_roll_joint
  [ 9] R_hip_yaw_joint
  [10] R_knee_pitch_joint
  [11] R_ankle_pitch_joint
  [12] R_ankle_roll_joint
----------------------------------------------------------------------
feet_indices              : [7, 13]
penalised_contact_indices : [4, 10, 5, 11]
termination_contact_indices: [0, 1]
ankle DOF indices (auto)  : [5, 6, 11, 12]
======================================================================
'train_cfg' provided -> Ignoring 'name=qub_flat'
Encoder MLP: Sequential(
  (0): Linear(in_features=510, out_features=256, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=256, out_features=128, bias=True)
  (3): ELU(alpha=1.0)
  (4): Linear(in_features=128, out_features=3, bias=True)
)
Actor MLP: Sequential(
  (0): Linear(in_features=57, out_features=512, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=512, out_features=256, bias=True)
  (3): ELU(alpha=1.0)
  (4): Linear(in_features=256, out_features=128, bias=True)
  (5): ELU(alpha=1.0)
  (6): Linear(in_features=128, out_features=13, bias=True)
)
Critic MLP: Sequential(
  (0): Linear(in_features=60, out_features=512, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=512, out_features=256, bias=True)
  (3): ELU(alpha=1.0)
  (4): Linear(in_features=256, out_features=128, bias=True)
  (5): ELU(alpha=1.0)
  (6): Linear(in_features=128, out_features=1, bias=True)
)
Loading model from: /home/kim/kudos_ws/rl_library/tron1-rl-isaacgym/logs/qub_flat/qub_flat/Sep04_00-19-14_/model_15000.pt
==============================================================
checkpoint : Sep04_00-19-14_/model_15000
cmd vx     : 0.4   측정 500 steps (10s)
alive      : 64/64
실제 vx    : +0.3742 (명령 0.4, 오차 6.5%)
--------------------------------------------------------------
foot0: mean=+0.0017  max_fwd=+0.0638  min_back=-0.0756  stride=0.1394
foot1: mean=+0.0030  max_fwd=+0.0678  min_back=-0.0850  stride=0.1528
--------------------------------------------------------------
좌우 전후위치 차이 : 0.13 cm
좌우 보폭 차이     : 1.34 cm (8.8%)
==============================================================
```

