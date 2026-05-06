import mujoco
import numpy as np
import rerun as rr
import time

from src.envs.acrobot import Acrobot
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig


def main(
    episodes: int = 5,
    steps: int = 1000,
    render: bool = True,
    out: str = "acrobot_mppi.mp4",
    rerun_port: int = 10001,
) -> None:
    env = Acrobot()
    cfg = MPPIConfig.load("acrobot")
    controller = MPPI(env, cfg)

    renderer = mujoco.Renderer(env.model, height=480, width=640) if render else None
    frames = []

    rr.init("acrobot costs")
    grpc_url = rr.serve_grpc(grpc_port=rerun_port)
    print(f"\nrerun gRPC: {grpc_url}")
    print(f"  rerun --connect rerun+http://127.0.0.1:{rerun_port}/proxy\n")

    for episode in range(episodes):
        env.reset()
        controller.reset()
        print(f"episode={episode} qpos={env.data.qpos}")
        state = env.get_state()
        total_cost = 0.0
        for t in range(steps):
            action, info = controller.plan_step(state)
            _, cost, done, _ = env.step(action)
            total_cost += cost

            state = env.get_state()
            sensor = env.data.sensordata.copy()
            c = env.running_cost_components(state, action, sensor)
            wc = env.weighted_cost_components(c)
            metrics = env.task_metrics()

            rr.set_time("step", sequence=episode * steps + t)
            rr.log("cost/step_running", rr.Scalars(float(cost)))
            rr.log("cost/episode_total", rr.Scalars(float(total_cost)))
            rr.log("cost/rollout_min", rr.Scalars(float(info["cost_min"])))
            rr.log("cost/rollout_mean", rr.Scalars(float(info["cost_mean"])))
            rr.log("mppi/n_eff", rr.Scalars(float(info["n_eff"])))

            rr.log("running_raw/tip_dist", rr.Scalars(float(c.tip_dist)))
            rr.log("running_raw/target_reward", rr.Scalars(float(c.target_reward)))
            rr.log("running_raw/target_cost", rr.Scalars(float(c.target_cost)))

            rr.log("running_weighted/target", rr.Scalars(float(wc.target_cost)))
            rr.log("running_weighted/total", rr.Scalars(float(wc.total)))

            rr.log("task/tip_z", rr.Scalars(float(metrics["tip_z"])))
            rr.log("task/tip_dist", rr.Scalars(float(metrics["tip_dist"])))
            rr.log("task/target_reward", rr.Scalars(float(metrics["target_reward"])))
            rr.log("task/target_cost", rr.Scalars(float(metrics["target_cost"])))
            rr.log("task/qvel_norm", rr.Scalars(float(metrics["qvel_norm"])))
            rr.log("task/success", rr.Scalars(float(metrics["success"])))

            if renderer is not None:
                renderer.update_scene(env.data)
                frame = renderer.render().copy()
                frames.append(frame)
                rr.log("frame", rr.Image(frame).compress())

            if t % 100 == 0:
                print(
                    f"episode={episode} step={t:4d} cost={cost:8.3f} "
                    f"S_min={info['cost_min']:8.2f} S_mean={info['cost_mean']:8.2f} "
                    f"n_eff={info['n_eff']:6.2f} "
                    f"shoulder={env.data.qpos[0]:.2f}  elbow={env.data.qpos[1]:.2f}  "
                    f"tip_z={metrics['tip_z']:.2f} tip_dist={metrics['tip_dist']:.2f} "
                    f"qvel={metrics['qvel_norm']:.2f} success={metrics['success']} "
                    f"total_cost={total_cost:.2f}"
                )
            if done:
                break
        print(f"episode={episode} total_cost={total_cost:.2f}")

    if renderer is not None and frames:
        import mediapy
        mediapy.write_video(out, frames, fps=30 * 5)
        print(f"saved {out}")
        renderer.close()
    env.close()

    print("\nDone logging. gRPC server still serving. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
