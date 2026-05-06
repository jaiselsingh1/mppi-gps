import numpy as np
import mujoco

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
) -> None:
    env = Humanoid(target_speed=target_speed)
    cfg = MPPIConfig.load("humanoid")
    controller = MPPI(env, cfg)

    np.random.seed(0)
    env.reset()
    state = env.get_state()
    frames = []
    renderer = mujoco.Renderer(env.model, height=480, width=640) if render else None
    rr.init("humanoid costs")
    grpc_url = rr.serve_grpc(grpc_port=10000)
    print(f"\nrerun gRPC: {grpc_url}")
    print("  rerun --connect rerun+http://127.0.0.1:10000/proxy\n")

    for t in range(steps):

        nominal = None
        if use_pd_nominal:
            nominal = np.tile(env.pose_pd_action(kp=pd_kp, kd=pd_kd), (cfg.H, 1))
        action, info = controller.plan_step(state, nominal=nominal)
        _, cost, done, _ = env.step(action)
        state = env.get_state()
        
        c = env.running_cost_components(state, action)
        wc = env.weighted_cost_components(c)

        rr.set_time("step", sequence=t)

        rr.log("cost/step_running", rr.Scalars(float(cost)))
        rr.log("cost/rollout_min", rr.Scalars(float(info["cost_min"])))
        rr.log("cost/rollout_mean", rr.Scalars(float(info["cost_mean"])))

        rr.log("stand", rr.Scalars(float(c.standing_reward)))
        rr.log("task", rr.Scalars(float(c.task_cost)))
        rr.log("lateral", rr.Scalars(float(c.root_y ** 2)))
        rr.log("lateral_vel", rr.Scalars(float(c.vy ** 2)))
        rr.log("root_angvel", rr.Scalars(float(c.root_angvel_sq)))
        rr.log("posture", rr.Scalars(float(c.posture_sq)))
        rr.log("qvel_sq", rr.Scalars(float(c.qvel_sq)))
        rr.log("ctrl", rr.Scalars(float(c.ctrl_sq)))

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
            frames.append(renderer.render().copy())

        if t % 25 == 0:
            metrics = env.task_metrics()
            print(
                f"step={t:4d} cost={cost:8.3f} "
                f"S_min={info['cost_min']:8.3f} S_mean={info['cost_mean']:8.3f} "
                f"n_eff={info['n_eff']:6.2f} "
                f"x={metrics['x_pos']:7.3f} z={metrics['torso_height']:5.3f} "
                f"head={metrics['head_height']:5.3f} "
                f"vx={metrics['forward_vx']:6.3f} "
                f"upright={metrics['upright']:5.3f} healthy={metrics['healthy']}"
            )

        if done:
            break

    if renderer is not None:
        import mediapy

        mediapy.write_video(out, frames, fps=30)
        print(f"saved {out}")

    print("\nDone logging. gRPC server still serving. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    import tyro

    tyro.cli(main)
