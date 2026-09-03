"""발바닥이 '땅에' 평평하게 닿는지 측정 (월드 기준).

왜 이렇게 재는가: 골반 기준 pitch만 보면 두 가지를 놓친다.
  1) roll(옆으로 기울어짐) - pitch가 0이어도 발이 옆으로 기울면 모서리만 닿는다
  2) 기준 프레임 - 보행 중 골반이 앞으로 기울어 있어서, 골반 기준 pitch가 0이어도
     발바닥은 땅에 평평하지 않다. '땅에 닿는가'는 월드 기준 문제다.
  실제로 2026-09-03에 골반 기준 pitch -1.32도(평평해 보임)인 발의 땅 기준 기울기가
  9.93도였다. 이 스크립트는 그 착오를 잡기 위해 만들었다.

측정 방법: 발 좌표계에서 본 중력 방향(base의 projected_gravity와 같은 개념).
발바닥이 땅과 평행하면 [0, 0, -1]이 되고, x/y 성분이 곧 기울기다.
  x 성분 -> 앞뒤 기울기(발끝/뒤꿈치가 들림)
  y 성분 -> 좌우 기울기(안쪽/바깥쪽 모서리만 닿음)
  tilt = asin(sqrt(x^2+y^2)) = 발바닥과 땅 사이 각도

접촉 중일 때만 본다. 스윙 중에는 발을 들어야 정상이므로 평평할 이유가 없다.
"""
import os
import isaacgym  # noqa: F401
from isaacgym.torch_utils import quat_rotate_inverse
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry

CMD_VX = 0.4
SETTLE_STEPS = 200
MEASURE_STEPS = 600


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

    body_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
    foot_names = [body_names[i] for i in env.feet_indices.tolist()]
    n_feet = len(env.feet_indices)
    grav = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)

    last_contact = torch.zeros(env.num_envs, n_feet, dtype=torch.bool, device=env.device)
    td = [{"x": [], "y": []} for _ in range(n_feet)]   # 접지 순간
    st = [{"x": [], "y": []} for _ in range(n_feet)]   # 스탠스 중
    contact_frac = [[] for _ in range(n_feet)]
    alive_h = []

    for i in range(SETTLE_STEPS + MEASURE_STEPS):
        ramp = min(1.0, i / 100.0)   # 반드시 램프업. 정지에서 목표속도를 바로 때리면 넘어진다.
        with torch.no_grad():
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
        env.commands[:, 0] = CMD_VX * ramp
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        obs, _, _, _, obs_history, commands, _ = env.step(actions.detach())

        rb = env.rigid_body_state.view(env.num_envs, env.num_bodies, 13)
        contacts = env.contact_forces[:, env.feet_indices, 2] > 1.0
        touchdown = contacts & (~last_contact)
        last_contact = contacts.clone()
        if i < SETTLE_STEPS:
            continue

        alive_h.append((env.root_states[:, 2] > 0.55)
                       & (env.projected_gravity[:, 2] < -0.85))
        for f in range(n_feet):
            fq = rb[:, env.feet_indices[f], 3:7]
            pg = quat_rotate_inverse(fq, grav)      # 발 좌표계에서 본 중력
            contact_frac[f].append(contacts[:, f].float().mean())
            if touchdown[:, f].any():
                td[f]["x"].append(pg[touchdown[:, f], 0])
                td[f]["y"].append(pg[touchdown[:, f], 1])
            if contacts[:, f].any():
                st[f]["x"].append(pg[contacts[:, f], 0])
                st[f]["y"].append(pg[contacts[:, f], 1])

    d = torch.rad2deg
    n_alive = int(torch.stack(alive_h).all(dim=0).sum())
    print("=" * 70)
    print(f"checkpoint : {args.load_run}/model_{args.checkpoint}   cmd_vx={CMD_VX}")
    print(f"foot0={foot_names[0]}, foot1={foot_names[1]}   생존 {n_alive}/{env.num_envs}")
    print("발 좌표계에서 본 중력. 발바닥이 땅과 평행하면 x=y=0.")
    print("  x = 앞뒤 기울기   y = 좌우 기울기(모서리로 딛는 정도)   tilt = 전체 각도")
    print("-" * 70)
    for f in range(n_feet):
        for lbl, D in (("접지 순간", td[f]), ("스탠스 중", st[f])):
            x = torch.cat(D["x"]) if D["x"] else torch.zeros(1)
            y = torch.cat(D["y"]) if D["y"] else torch.zeros(1)
            tilt = torch.asin(torch.clip(torch.sqrt(x ** 2 + y ** 2), 0, 1))
            print(f"[foot{f} {foot_names[f]:14s}] {lbl}: "
                  f"x {d(torch.asin(torch.clip(x.mean(),-1,1))):+6.2f} deg  "
                  f"y {d(torch.asin(torch.clip(y.mean(),-1,1))):+6.2f} deg  "
                  f"tilt {d(tilt.mean()):5.2f} deg (sd {d(tilt.std()):.2f})")
        print(f"   접촉 시간 비율 : {torch.stack(contact_frac[f]).mean()*100:.1f}%")
        print("-" * 70)
    print("해석: tilt가 0에 가까울수록 발바닥 전체가 땅에 닿는다.")
    print("=" * 70)


if __name__ == "__main__":
    main(get_args())
