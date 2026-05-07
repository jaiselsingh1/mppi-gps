import numpy as np
import mujoco
from pathlib import Path
from typing import Literal
from uuid import uuid4

from src.envs.humanoid import Humanoid
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
import rerun as rr
import time


def main(
    steps: int = 500,
    target_speed: float = 0.0,
    use_pd_nominal: bool = False,
    pd_kp: float = 20.0,
    pd_kd: float = 5.0,
    render: bool = True,
    out: str = "humanoid_mppi.mp4",
    rerun_port: int = 10000,
    rerun_mode: Literal["serve", "spawn", "save", "off"] = "serve",
    rerun_path: str = "runs/humanoid_latest.rrd",
    stop_on_unhealthy: bool = False,
    terminal_stand_weight: float = 48.0,
    K: int | None = None,
    H: int | None = None,
    lam: float | None = None,
    noise_sigma: float | None = None,
    clip_actions: bool = False,
) -> None:
    env = Humanoid(
        target_speed=target_speed,
        terminal_stand_weight=terminal_stand_weight,
    )
    cfg = MPPIConfig.load("humanoid")
    if K is not None:
        cfg.K = K
    if H is not None:
        cfg.H = H
    if lam is not None:
        cfg.lam = lam
    if noise_sigma is not None:
        cfg.noise_sigma = noise_sigma
    cfg.clip_actions = clip_actions
    controller = MPPI(env, cfg)
    task_name = "walk" if target_speed > 0.0 else "stand"
    print(
        f"humanoid task: {task_name} target_speed={target_speed} "
        f"terminal_stand_weight={terminal_stand_weight}"
    )

    np.random.seed(0)
    env.reset()
    state = env.get_state()
    frames = []
    renderer = mujoco.Renderer(env.model, height=480, width=640) if render else None
    use_rerun = rerun_mode != "off"
    if use_rerun:
        rr.init("humanoid costs", recording_id=f"humanoid-{uuid4()}")
        if rerun_mode == "serve":
            grpc_url = rr.serve_grpc(grpc_port=rerun_port)
            print(f"\nrerun gRPC: {grpc_url}")
            print(f"  rerun --connect rerun+http://127.0.0.1:{rerun_port}/proxy")
            print(f"  over SSH: ssh -L {rerun_port}:127.0.0.1:{rerun_port} <host>\n", flush=True)
        elif rerun_mode == "spawn":
            rr.spawn(port=rerun_port)
            print(f"\nrerun viewer spawned on port {rerun_port}\n", flush=True)
        elif rerun_mode == "save":
            Path(rerun_path).parent.mkdir(parents=True, exist_ok=True)
            rr.save(rerun_path)
            print(f"\nrerun recording: {rerun_path}", flush=True)

        rr.set_time("step", sequence=0)
        rr.log("status/started", rr.Scalars(1.0))
        if renderer is not None:
            renderer.update_scene(env.data)
            initial_frame = renderer.render().copy()
            rr.log("frame", rr.Image(initial_frame).compress())

    for t in range(steps):

        nominal = None
        if use_pd_nominal:
            nominal = np.tile(env.pose_pd_action(kp=pd_kp, kd=pd_kd), (cfg.H, 1))
        action, info = controller.plan_step(state, nominal=nominal)
        _, cost, done, _ = env.step(action)
        state = env.get_state()
        metrics = env.task_metrics()

        c = env.running_cost_components(state, action, env.data.sensordata.copy())
        wc = env.weighted_cost_components(c)

        if use_rerun:
            rr.set_time("step", sequence=t)

            rr.log("cost/step_running", rr.Scalars(float(cost)))
            rr.log("cost/rollout_min", rr.Scalars(float(info["cost_min"])))
            rr.log("cost/rollout_mean", rr.Scalars(float(info["cost_mean"])))
            rr.log("mppi/n_eff", rr.Scalars(float(info["n_eff"])))

            rr.log("stand", rr.Scalars(float(c.standing_reward)))
            rr.log("task", rr.Scalars(float(c.task_cost)))
            rr.log("reward/task", rr.Scalars(float(c.task_reward)))
            rr.log("reward/small_control", rr.Scalars(float(c.small_control)))
            rr.log("cost/reward", rr.Scalars(float(c.reward_cost)))
            rr.log("lateral", rr.Scalars(float(c.root_y ** 2)))
            rr.log("lateral_vel", rr.Scalars(float(c.vy ** 2)))
            rr.log("root_angvel", rr.Scalars(float(c.root_angvel_sq)))
            rr.log("posture", rr.Scalars(float(c.posture_sq)))
            rr.log("qvel_sq", rr.Scalars(float(c.qvel_sq)))
            rr.log("ctrl", rr.Scalars(float(c.ctrl_sq)))
            rr.log("action/absmax", rr.Scalars(float(np.max(np.abs(action)))))

            rr.log("weighted_cost/reward", rr.Scalars(float(wc.reward_cost)))
            rr.log("weighted_cost/stand", rr.Scalars(float(wc.stand_cost)))
            rr.log("weighted_cost/task", rr.Scalars(float(wc.task_cost)))
            rr.log("weighted_cost/lateral", rr.Scalars(float(wc.lateral_cost)))
            rr.log("weighted_cost/lateral_vel", rr.Scalars(float(wc.lateral_vel_cost)))
            rr.log("weighted_cost/root_angvel", rr.Scalars(float(wc.root_angvel_cost)))
            rr.log("weighted_cost/posture", rr.Scalars(float(wc.posture_cost)))
            rr.log("weighted_cost/qvel", rr.Scalars(float(wc.qvel_cost)))
            rr.log("weighted_cost/ctrl", rr.Scalars(float(wc.ctrl_cost)))
            rr.log("weighted_cost/total", rr.Scalars(float(wc.total)))

    
        if renderer is not None:
            renderer.update_scene(env.data)
            frame = renderer.render().copy()
            frames.append(frame)
            if use_rerun:
                rr.log("frame", rr.Image(frame).compress())

        if t % 25 == 0:
            print(
                f"step={t:4d} cost={cost:8.3f} "
                f"S_min={info['cost_min']:8.3f} S_mean={info['cost_mean']:8.3f} "
                f"n_eff={info['n_eff']:6.2f} "
                f"x={metrics['x_pos']:7.3f} z={metrics['torso_height']:5.3f} "
                f"head={metrics['head_height']:5.3f} "
                f"vx={metrics['forward_vx']:6.3f} "
                f"upright={metrics['upright']:5.3f} "
                f"stand={float(c.standing_reward):5.3f} "
                f"task={float(c.task_reward):5.3f} "
                f"small_ctrl={float(c.small_control):5.3f} "
                f"a_max={np.max(np.abs(action)):5.3f} "
                f"healthy={metrics['healthy']}"
            )
            if use_rerun:
                rr.log("vx", rr.Scalars(float(metrics["forward_vx"])))

        if stop_on_unhealthy and not metrics["healthy"]:
            print(f"stopping: unhealthy at step={t}")
            break
        if done:
            break

    if renderer is not None:
        import mediapy

        mediapy.write_video(out, frames, fps=30)
        print(f"saved {out}")

    if rerun_mode == "serve":
        print("\nDone logging. gRPC server still serving. Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    import tyro

    tyro.cli(main)
