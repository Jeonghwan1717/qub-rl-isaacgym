"""두 발이 실제로 얼마나 가까워지는지 측정 — 특히 뒤꿈치 충돌 여유.

왜 필요한가: hip_yaw로 발끝이 바깥을 향하면 뒤꿈치는 반대로 안쪽을 향한다.
발 중심 사이 거리(feet_distance_max가 보는 값)는 멀어도 뒤꿈치 모서리끼리는
가까워질 수 있다. '부딪히느냐'는 중심 거리가 아니라 모서리 거리 문제다.

발 충돌 형상은 단순 박스다(STL 삼각형 12개):
    앞뒤 -0.090 ~ +0.110 m   좌우 -0.045 ~ +0.045 m
즉 뒤꿈치는 발 링크 원점보다 9cm 뒤, 발 폭은 9cm.

박스 바닥 네 모서리를 월드로 변환해 두 발 사이 최소 거리를 구한다.
뒤꿈치 안쪽 모서리(왼발 -y / 오른발 +y 쪽 뒤 모서리)는 따로도 본다 —
발끝이 벌어질 때 가장 먼저 만나는 지점이 거기다.
"""
import os
import isaacgym  # noqa: F401
from isaacgym.torch_utils import quat_apply
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry

CMD_VX = 0.4
SETTLE_STEPS = 200
MEASURE_STEPS = 600

# 발 충돌 박스 (STL 실측). 바닥면 네 모서리를 쓴다.
X_BACK, X_FRONT = -0.090, 0.110
Y_HALF = 0.045
Z_SOLE = -0.038


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
    fnames = [body_names[i] for i in env.feet_indices.tolist()]
    li = fnames.index("L_foot_link")
    ri = fnames.index("R_foot_link")
    Lb = int(env.feet_indices[li])
    Rb = int(env.feet_indices[ri])

    dev = env.device
    # 바닥 네 모서리 (뒤안쪽, 뒤바깥, 앞안쪽, 앞바깥) — 안/바깥은 발마다 부호가 다르다.
    # 왼발은 -y가 안쪽(오른발 쪽), 오른발은 +y가 안쪽.
    corners_L = torch.tensor([[X_BACK, -Y_HALF, Z_SOLE], [X_BACK, +Y_HALF, Z_SOLE],
                              [X_FRONT, -Y_HALF, Z_SOLE], [X_FRONT, +Y_HALF, Z_SOLE]],
                             device=dev)
    corners_R = torch.tensor([[X_BACK, +Y_HALF, Z_SOLE], [X_BACK, -Y_HALF, Z_SOLE],
                              [X_FRONT, +Y_HALF, Z_SOLE], [X_FRONT, -Y_HALF, Z_SOLE]],
                             device=dev)

    min_any, heel_inner, center_d, alive_h = [], [], [], []

    for i in range(SETTLE_STEPS + MEASURE_STEPS):
        ramp = min(1.0, i / 100.0)
        with torch.no_grad():
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
        env.commands[:, 0] = CMD_VX * ramp
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        obs, _, _, _, obs_history, commands, _ = env.step(actions.detach())
        if i < SETTLE_STEPS:
            continue

        rb = env.rigid_body_state.view(env.num_envs, env.num_bodies, 13)
        out = []
        for body, corn in ((Lb, corners_L), (Rb, corners_R)):
            p = rb[:, body, 0:3]
            q = rb[:, body, 3:7]
            pts = []
            for c in corn:
                cc = c.unsqueeze(0).repeat(env.num_envs, 1)
                pts.append(p + quat_apply(q, cc))
            out.append(torch.stack(pts, dim=1))          # (env, 4, 3)
        PL, PR = out
        # 모든 모서리 쌍의 수평 거리 중 최소
        d = torch.norm(PL[:, :, None, :2] - PR[:, None, :, :2], dim=-1)   # (env,4,4)
        min_any.append(d.reshape(env.num_envs, -1).min(dim=1)[0])
        # 뒤꿈치 안쪽 모서리끼리 (각 배열의 0번)
        heel_inner.append(torch.norm(PL[:, 0, :2] - PR[:, 0, :2], dim=-1))
        center_d.append(torch.norm(rb[:, Lb, :2] - rb[:, Rb, :2], dim=-1))
        alive_h.append((env.root_states[:, 2] > 0.55)
                       & (env.projected_gravity[:, 2] < -0.85))

    keep = torch.stack(alive_h).all(dim=0)
    n = int(keep.sum())
    MA = torch.stack(min_any)[:, keep]
    HI = torch.stack(heel_inner)[:, keep]
    CD = torch.stack(center_d)[:, keep]

    print("=" * 68)
    print(f"checkpoint : {args.load_run}/model_{args.checkpoint}   cmd_vx={CMD_VX}")
    print(f"생존 {n}/{env.num_envs}   발 박스 앞뒤 {X_BACK:+.3f}~{X_FRONT:+.3f} / "
          f"폭 {2*Y_HALF:.3f} m")
    print("-" * 68)
    print(f"발 중심 거리        평균 {CD.mean()*100:6.2f} cm   최소 {CD.min()*100:6.2f} cm")
    print(f"뒤꿈치 안쪽 모서리   평균 {HI.mean()*100:6.2f} cm   최소 {HI.min()*100:6.2f} cm")
    print(f"두 발 최소 간격      평균 {MA.mean()*100:6.2f} cm   "
          f"최소 {MA.min()*100:6.2f} cm   <- 충돌 여유")
    print("-" * 68)
    worst = float(MA.min()) * 100
    if worst <= 0:
        print(f"!! 겹침 발생 (최소 {worst:.2f} cm) - 실제로 부딪힌다")
    elif worst < 2.0:
        print(f"!! 여유 {worst:.2f} cm - 매우 아슬아슬하다")
    elif worst < 5.0:
        print(f"주의: 여유 {worst:.2f} cm - 여유가 크지 않다")
    else:
        print(f"여유 {worst:.2f} cm - 충돌 위험 없음")
    print("=" * 68)


if __name__ == "__main__":
    main(get_args())
