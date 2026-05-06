import numpy as np
import mujoco

from src.envs.humanoid import Humanoid
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
import rerun as rr 


def main(
    steps: int = 500,
    target_speed: float = 0.0,
    use_pd_nominal: bool = True,
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

    try:
        for t in range(steps):
            nominal = None
            if use_pd_nominal:
                nominal = np.tile(env.pose_pd_action(kp=pd_kp, kd=pd_kd), (cfg.H, 1))
            action, info = controller.plan_step(state, nominal=nominal)
            _, cost, done, _ = env.step(action)
            state = env.get_state()

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
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
