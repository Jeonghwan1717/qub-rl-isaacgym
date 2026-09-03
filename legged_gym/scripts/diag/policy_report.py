"""한 체크포인트의 보행 지표를 '한 번의 시뮬레이션'으로 전부 뽑는다.

개별 진단 스크립트(foot_flatness / joint_sym_check / step_asym / heel_clearance)를
따로 돌리면 같은 시뮬레이션을 네 번 반복하게 된다. 체크포인트를 여러 개 훑을 때는
그 비용이 커서, 한 루프에서 모든 지표를 모으도록 합쳤다.
계산 내용과 근거는 각 개별 스크립트의 docstring과 동일하다.

--tsv 를 주면 한 줄 TSV로만 출력한다(여러 체크포인트를 표로 모을 때 사용).
"""
import argparse
import os
import isaacgym  # noqa: F401
from isaacgym.torch_utils import (quat_mul, quat_conjugate, get_euler_xyz,
                                  quat_rotate_inverse, quat_apply)
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry

CMD_VX = 0.4
SETTLE_STEPS = 200
MEASURE_STEPS = 500
PELVIS_BODY_IDX = 1

# 발 충돌 박스 (STL 실측): 앞뒤 -0.090~+0.110, 폭 ±0.045
X_BACK, X_FRONT, Y_HALF, Z_SOLE = -0.090, 0.110, 0.045, -0.038

PAIRS = [("hip_pitch",   "L_hip_pitch_joint",   "R_hip_pitch_joint",   +1),
         ("hip_roll",    "L_hip_roll_joint",    "R_hip_roll_joint",    -1),
         ("hip_yaw",     "L_hip_yaw_joint",     "R_hip_yaw_joint",     -1),
         ("knee_pitch",  "L_knee_pitch_joint",  "R_knee_pitch_joint",  +1),
         ("ankle_pitch", "L_ankle_pitch_joint", "R_ankle_pitch_joint", -1),
         ("ankle_roll",  "L_ankle_roll_joint",  "R_ankle_roll_joint",  -1)]


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def main(args, tsv):
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

    dev = env.device
    names = env.dof_names
    idx = [(names.index(l), names.index(r), m) for _, l, r, m in PAIRS]
    tyaw_dof = names.index("torso_yaw_joint")
    bn = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
    fn = [bn[i] for i in env.feet_indices.tolist()]
    li, ri = fn.index("L_foot_link"), fn.index("R_foot_link")
    Lb, Rb = int(env.feet_indices[li]), int(env.feet_indices[ri])
    grav = torch.tensor([0.0, 0.0, -1.0], device=dev).repeat(env.num_envs, 1)
    cornL = torch.tensor([[X_BACK, -Y_HALF, Z_SOLE], [X_BACK, Y_HALF, Z_SOLE],
                          [X_FRONT, -Y_HALF, Z_SOLE], [X_FRONT, Y_HALF, Z_SOLE]], device=dev)
    cornR = torch.tensor([[X_BACK, Y_HALF, Z_SOLE], [X_BACK, -Y_HALF, Z_SOLE],
                          [X_FRONT, Y_HALF, Z_SOLE], [X_FRONT, -Y_HALF, Z_SOLE]], device=dev)

    tilt_st = {Lb: [], Rb: []}
    cfrac = {Lb: [], Rb: []}
    jl, jr, tyaw_h, footx_h, vx_h, gap_h, alive_h = [], [], [], [], [], [], []

    for i in range(SETTLE_STEPS + MEASURE_STEPS):
        ramp = min(1.0, i / 100.0)   # 반드시 램프업 (정지→목표속도 즉시 인가는 넘어짐 유발)
        with torch.no_grad():
            est = encoder(obs_history)
            act = policy(torch.cat((est, obs, commands), dim=-1).detach())
        env.commands[:, 0] = CMD_VX * ramp
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        obs, _, _, _, obs_history, commands, _ = env.step(act.detach())
        if i < SETTLE_STEPS:
            continue

        rb = env.rigid_body_state.view(env.num_envs, env.num_bodies, 13)
        # 발바닥 기울기 (접촉 중)
        for b in (Lb, Rb):
            c = env.contact_forces[:, b, 2] > 1.0
            cfrac[b].append(c.float().mean())
            if c.any():
                pg = quat_rotate_inverse(rb[:, b, 3:7], grav)
                tilt_st[b].append(torch.asin(torch.clip(
                    torch.norm(pg[:, :2], dim=1), 0, 1))[c])
        # 뒤꿈치 여유
        pts = []
        for b, cn in ((Lb, cornL), (Rb, cornR)):
            p, q = rb[:, b, 0:3], rb[:, b, 3:7]
            pts.append(torch.stack(
                [p + quat_apply(q, c.unsqueeze(0).repeat(env.num_envs, 1)) for c in cn], 1))
        d = torch.norm(pts[0][:, :, None, :2] - pts[1][:, None, :, :2], dim=-1)
        gap_h.append(d.reshape(env.num_envs, -1).min(dim=1)[0])
        # 관절 대칭
        jl.append(torch.stack([env.dof_pos[:, a] for a, _, _ in idx], 1))
        jr.append(torch.stack([env.dof_pos[:, b] for _, b, _ in idx], 1))
        tyaw_h.append(env.dof_pos[:, tyaw_dof].clone())
        # 발 앞뒤 위치 (골반 기준)
        pq = rb[:, PELVIS_BODY_IDX, 3:7]
        rel = env.foot_positions - rb[:, PELVIS_BODY_IDX, 0:3].unsqueeze(1)
        qq = pq.repeat_interleave(len(env.feet_indices), dim=0)
        footx_h.append(quat_rotate_inverse(qq, rel.reshape(-1, 3)).reshape(
            env.num_envs, len(env.feet_indices), 3)[:, :, 0])
        vx_h.append(env.base_lin_vel[:, 0].clone())
        alive_h.append((env.root_states[:, 2] > 0.55)
                       & (env.projected_gravity[:, 2] < -0.85))

    keep = torch.stack(alive_h).all(dim=0)
    n = int(keep.sum())
    if n == 0:
        print(f"{args.checkpoint}\t0\t-\t-\t-\t-\t-\t-\t-\t-\t-")
        return

    d2 = torch.rad2deg
    tL = float(d2(torch.cat(tilt_st[Lb]).mean()))
    tR = float(d2(torch.cat(tilt_st[Rb]).mean()))
    JL = torch.stack(jl)[:, keep]
    JR = torch.stack(jr)[:, keep]
    mask = torch.tensor([m for _, _, m in idx], dtype=torch.float, device=dev)
    res = (JL - mask.view(1, 1, -1) * JR).mean(dim=0)          # (env, 6)
    per_joint = (res ** 2).mean(dim=0)                          # (6,)
    jsum = float(per_joint.sum())
    jmax = float(d2(res.mean(dim=0).abs().max()))
    X = torch.stack(footx_h)[:, keep, :]
    m0, m1 = X[:, :, 0].mean().item(), X[:, :, 1].mean().item()
    s0 = (X[:, :, 0].max(0)[0] - X[:, :, 0].min(0)[0]).mean().item()
    s1 = (X[:, :, 1].max(0)[0] - X[:, :, 1].min(0)[0]).mean().item()
    vx = float(torch.stack(vx_h)[:, keep].mean())
    gap = float(torch.stack(gap_h)[:, keep].min())
    ty = float(d2(torch.stack(tyaw_h)[:, keep].mean()))

    row = (f"{args.checkpoint}\t{n}/{env.num_envs}\t{vx:.3f}\t"
           f"{abs(vx-CMD_VX)/CMD_VX*100:.1f}\t{tL:.2f}\t{tR:.2f}\t"
           f"{jsum:.5f}\t{jmax:.2f}\t{abs(m0-m1)*100:.2f}\t"
           f"{abs(s0-s1)/max(s0, s1)*100:.1f}\t{gap*100:.2f}\t{ty:+.2f}")
    if tsv:
        print(row)
    else:
        print("=" * 60)
        print(f"checkpoint {args.load_run}/model_{args.checkpoint}")
        print(f"생존 {n}/{env.num_envs}   실제 vx {vx:.3f} (오차 "
              f"{abs(vx-CMD_VX)/CMD_VX*100:.1f}%)")
        print(f"발바닥 기울기   L {tL:.2f}°   R {tR:.2f}°")
        print(f"6관절 잔차 합   {jsum:.5f} rad²   최대 관절 {jmax:.2f}°")
        print(f"발 앞뒤 차이    {abs(m0-m1)*100:.2f} cm   "
              f"보폭 차이 {abs(s0-s1)/max(s0,s1)*100:.1f}%")
        print(f"뒤꿈치 최소 여유 {gap*100:.2f} cm   torso_yaw DC {ty:+.2f}°")
        print("=" * 60)


if __name__ == "__main__":
    tsv = "--tsv" in os.sys.argv
    if tsv:
        os.sys.argv.remove("--tsv")
    main(get_args(), tsv)
