"""다리 6관절의 좌우 미러 대칭 잔차 + torso_yaw + 발 heading 측정.

미러 부호는 URDF 축에서 직접 유도했다. 시상면 반사에서 각속도는 유사벡터라
(wx,wy,wz) -> (-wx,+wy,-wz), 즉 pitch(y)는 부호 유지, roll(x)/yaw(z)는 반전.
좌우 관절 축이 서로 반대이면 거기서 한 번 더 반전된다.
QUB는 12개 다리 관절 origin이 전부 rpy="0 0 0"이라 링크 프레임이 모두 베이스와
정렬돼 있어 URDF에 적힌 축이 곧 월드 축이다.
  hip_pitch   R(0,1,0)  L(0,1,0)   같음 -> +1 (대칭 조건 L=R)
  hip_roll    R(1,0,0)  L(1,0,0)   같음 -> -1 (L=-R)
  hip_yaw     R(0,0,1)  L(0,0,1)   같음 -> -1
  knee_pitch  R(0,-1,0) L(0,-1,0)  같음 -> +1
  ankle_pitch R(0,-1,0) L(0,+1,0)  반대 -> -1   <- 유일하게 축이 반대
  ankle_roll  R(1,0,0)  L(1,0,0)   같음 -> -1
잔차 = L - mask*R 로 통일된다.

DC/AC를 나누는 이유: 보행은 좌우 180도 위상차라 매 순간 좌우가 다른 게 정상이다.
문제는 한쪽으로 치우쳐 '고정된' 성분(DC)이다. 시정수 2초 평균으로 분리한다.
"""
import os
import isaacgym  # noqa: F401
from isaacgym.torch_utils import quat_mul, quat_conjugate, get_euler_xyz
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry

CMD_VX = 0.4
SETTLE_STEPS = 200
MEASURE_STEPS = 500
PELVIS_BODY_IDX = 1

PAIRS = [("hip_pitch",   "L_hip_pitch_joint",   "R_hip_pitch_joint",   +1),
         ("hip_roll",    "L_hip_roll_joint",    "R_hip_roll_joint",    -1),
         ("hip_yaw",     "L_hip_yaw_joint",     "R_hip_yaw_joint",     -1),
         ("knee_pitch",  "L_knee_pitch_joint",  "R_knee_pitch_joint",  +1),
         ("ankle_pitch", "L_ankle_pitch_joint", "R_ankle_pitch_joint", -1),
         ("ankle_roll",  "L_ankle_roll_joint",  "R_ankle_roll_joint",  -1)]


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def main(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 64
    env_cfg.env.episode_length_s = 1000
    env_cfg.noise.add_noise = False
    for k in ("randomize_friction", "randomize_restitution", "randomize_base_com",
              "push_robots", "randomize_Kp", "randomize_Kd", "randomize_motor_torque",
              "randomize_default_dof_pos", "randomize_action_delay"):
        setattr(env_cfg.domain_rand, k, False)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, obs_history, commands, _ = env.get_observations()

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)
    encoder = runner.get_inference_encoder(device=env.device)

    names = env.dof_names
    idx = {lbl: (names.index(ln), names.index(rn), m) for lbl, ln, rn, m in PAIRS}
    tyaw_dof = names.index("torso_yaw_joint")
    body_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
    foot_names = [body_names[i] for i in env.feet_indices.tolist()]

    rec = {lbl: {"L": [], "R": []} for lbl in idx}
    tyaw_h, fy0, fy1, alive_h = [], [], [], []

    for i in range(SETTLE_STEPS + MEASURE_STEPS):
        ramp = min(1.0, i / 100.0)   # 반드시 램프업. 정지에서 목표속도를 바로 때리면 넘어진다.
        with torch.no_grad():
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
        env.commands[:, 0] = CMD_VX * ramp
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        obs, _, _, _, obs_history, commands, _ = env.step(actions.detach())
        if i < SETTLE_STEPS:
            continue

        for lbl, (iL, iR, _m) in idx.items():
            rec[lbl]["L"].append(env.dof_pos[:, iL].clone())
            rec[lbl]["R"].append(env.dof_pos[:, iR].clone())
        tyaw_h.append(env.dof_pos[:, tyaw_dof].clone())
        rb = env.rigid_body_state.view(env.num_envs, env.num_bodies, 13)
        pq = rb[:, PELVIS_BODY_IDX, 3:7]
        for lst, f in ((fy0, 0), (fy1, 1)):
            _, _, y = get_euler_xyz(quat_mul(quat_conjugate(pq),
                                             rb[:, env.feet_indices[f], 3:7]))
            lst.append(wrap(y))
        alive_h.append((env.root_states[:, 2] > 0.55)
                       & (env.projected_gravity[:, 2] < -0.85))

    keep = torch.stack(alive_h).all(dim=0)
    n_keep = int(keep.sum())
    d = torch.rad2deg
    print("=" * 72)
    print(f"checkpoint : {args.load_run}/model_{args.checkpoint}   cmd_vx={CMD_VX}")
    print(f"alive      : {n_keep}/{env.num_envs}   "
          f"foot0={foot_names[0]}, foot1={foot_names[1]}")
    if n_keep == 0:
        print("살아남은 개체 없음 - 측정 불가")
        return

    T = torch.stack(tyaw_h)[:, keep]
    print("-" * 72)
    print(f"[torso_yaw]  DC {d(T.mean(dim=0).mean()):+6.2f} deg   "
          f"AC {d(T.std(dim=0).mean()):5.2f} deg")

    total = 0.0
    for lbl, (iL, iR, m) in idx.items():
        L = torch.stack(rec[lbl]["L"])[:, keep]
        R = torch.stack(rec[lbl]["R"])[:, keep]
        mres = (L - m * R).mean(dim=0)
        sq = float((mres ** 2).mean())
        total += sq
        print("-" * 72)
        print(f"[{lbl}]  mask {m:+d}  (대칭 조건 L {'=' if m > 0 else '= -'} R)")
        print(f"  L: DC {d(L.mean(dim=0).mean()):+7.2f}  AC {d(L.std(dim=0).mean()):5.2f}"
              f"   R: DC {d(R.mean(dim=0).mean()):+7.2f}  AC {d(R.std(dim=0).mean()):5.2f}")
        print(f"  >> 미러 잔차 {d(mres.mean()):+7.2f} deg    제곱 {sq:.5f} rad^2")

    F0 = torch.stack(fy0)[:, keep]
    F1 = torch.stack(fy1)[:, keep]
    print("-" * 72)
    print(f"[6관절 제곱합] {total:.5f} rad^2   "
          f"(= _reward_joint_symmetry의 raw. scale -20이면 로그에 {total*20:.4f})")
    print(f"[발 heading]  foot0 {d(F0.mean(dim=0).mean()):+6.2f}  "
          f"foot1 {d(F1.mean(dim=0).mean()):+6.2f}  "
          f"잔차 {d((F0 + F1).mean(dim=0).mean()):+.2f} deg")
    print("=" * 72)


if __name__ == "__main__":
    main(get_args())
