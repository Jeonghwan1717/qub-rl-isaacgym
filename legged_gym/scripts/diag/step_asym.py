"""좌우 발의 '몸 기준 앞뒤 위치'와 보폭 비대칭 측정.

기준 프레임은 base_link가 아니라 pelvis_link다. 다리는 pelvis에 달려 있고 그 사이에
torso_yaw_joint가 있어서, base 기준으로 재면 골반을 비트는 것만으로 좌우 오프셋이
전후축에 섞여 수치가 실제보다 좋아진다(2026-09-02 실측: base 5.72cm vs pelvis 7.02cm).

⚠️ 명령은 반드시 램프업한다. 정지 상태에서 목표 속도를 바로 때리면 넘어지는 개체가
생겨 통계가 오염된다(0.6 스윕에서 생존 74% vs 램프 시 100%).
"""
import os
import isaacgym  # noqa: F401
from isaacgym.torch_utils import quat_rotate_inverse
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry

CMD_VX = 0.4
SETTLE_STEPS = 200
MEASURE_STEPS = 500
PELVIS_BODY_IDX = 1


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

    n_feet = len(env.feet_indices)
    hist, alive_hist, vx_hist = [], [], []

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
        pelv_pos = rb[:, PELVIS_BODY_IDX, 0:3]
        pelv_quat = rb[:, PELVIS_BODY_IDX, 3:7]
        rel = env.foot_positions - pelv_pos.unsqueeze(1)
        q = pelv_quat.repeat_interleave(n_feet, dim=0)
        foot_x = quat_rotate_inverse(
            q, rel.reshape(-1, 3)).reshape(env.num_envs, n_feet, 3)[:, :, 0]
        hist.append(foot_x.clone())
        vx_hist.append(env.base_lin_vel[:, 0].clone())
        alive_hist.append((env.root_states[:, 2] > 0.55)
                          & (env.projected_gravity[:, 2] < -0.85))

    A = torch.stack(alive_hist)
    keep = A.all(dim=0)
    n_keep = int(keep.sum())
    print("=" * 62)
    print(f"checkpoint : {args.load_run}/model_{args.checkpoint}")
    print(f"cmd vx     : {CMD_VX}   측정 {MEASURE_STEPS} steps "
          f"({MEASURE_STEPS*0.02:.0f}s)")
    print(f"alive      : {n_keep}/{env.num_envs}")
    if n_keep == 0:
        print("살아남은 개체 없음 - 측정 불가")
        return

    X = torch.stack(hist)[:, keep, :]
    VX = torch.stack(vx_hist)[:, keep]
    print(f"실제 vx    : {VX.mean():+.4f} (명령 {CMD_VX}, "
          f"오차 {abs(VX.mean()-CMD_VX)/CMD_VX*100:.1f}%)")
    print("-" * 62)
    means, strides = [], []
    for f in range(X.shape[2]):
        xf = X[:, :, f]
        mean = xf.mean().item()
        # 개체별 최대/최소를 먼저 낸 뒤 평균 — 전체 평균의 max를 쓰면 위상이 섞여 뭉개진다
        fwd = xf.max(dim=0)[0].mean().item()
        back = xf.min(dim=0)[0].mean().item()
        means.append(mean)
        strides.append(fwd - back)
        print(f"foot{f}: mean={mean:+.4f}  max_fwd={fwd:+.4f}  "
              f"min_back={back:+.4f}  stride={fwd-back:.4f}")
    print("-" * 62)
    print(f"좌우 전후위치 차이 : {abs(means[0]-means[1])*100:.2f} cm")
    sd = abs(strides[0] - strides[1]) / max(strides) * 100
    print(f"좌우 보폭 차이     : {abs(strides[0]-strides[1])*100:.2f} cm ({sd:.1f}%)")
    print("=" * 62)


if __name__ == "__main__":
    main(get_args())
