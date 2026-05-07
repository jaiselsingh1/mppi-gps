import time 
import numpy as np
import warp as wp
import mujoco 
import mujoco_warp as mjw 
from mujoco import rollout 
import torch 

from src.envs.acrobot import Acrobot 

@wp.kernel 
def sample_actions(
    actions: wp.array3d(dtype = wp.float32), 
    seed: int, 
    sigma: float, 
):
    h, k, a = wp.tid()
    rng = wp.rand_init(seed, h * actions.shape[1] + k)
    u1 = wp.randf(rng)
    u2 = wp.randf(rng)
    if u1 < 1.0e-6:
        u1 = 1.0e-6
    z = wp.sqrt(-2.0 * wp.log(u1)) * wp.cos(6.28318530718 * u2)
    actions[h, k, a] = sigma * z

@wp.kernel 
def zero_costs(costs: wp.array(dtype = wp.float32)):
    k = wp.tid()
    costs[k] = 0.0 

@wp.kernel
def add_acrobot_cost(
    sensordata: wp.array2d(dtype = wp.float32), 
    costs: wp.array(dtype = wp.float32)):

    k = wp.tid()
    dx = sensordata[k, 0]
    dy = sensordata[k, 1]
    dz = sensordata[k, 2] - 4.0
    dist = wp.sqrt(dx * dx + dy * dy + dz * dz)

    reward = 1.0
    if dist > 0.2:
        d = dist - 0.2
        reward = wp.exp(wp.log(0.1) * d * d)

    costs[k] += 1.0 - reward

def acrobot_cost_numpy(sensordata):
    tip = sensordata[..., :3]
    dist = np.linalg.norm(tip - np.array([0.0, 0.0, 0.0, 4.0]), axis = -1)
    reward = np.where(dist <= 0.2, 1.0, np.exp(np.log(0.1) * (dist - 0.2) ** 2))
    return (1.0 - reward).sum(axis=1)

def bench_cpu(
        K = 512, 
        H = 256, 
        n_batches = 10, 
        n_timed = 20, 
        sigma = 1.5
):
    env = Acrobot()
    ctx = rollout.Rollout(nthread=32)
    data_pool = [mujoco.MjData(env.model) for _ in range(32)]

    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        s0 = env.get_state()
        actions = np.random.normal(0.0, sigma, (n_batches * K, H, env.action_dim))
        states, sensordata = ctx.rollout(env.model, data_pool, s0, actions)

        times.append(time.perf_counter() - t0)
    

    print("cpu sample + rollout + cost", np.mean(times), "s")
    env.close()
    return times 


def bench_warp(
    K = 512, 
    H = 256, 
    n_batches = 10, 
    n_timed = 20, 
    sigma = 1.5, 
    device = "cuda", 
):
    nworld = n_batches * K 
    env = Acrobot(use_warp = True, nworld = nworld)

    env.reset()
    qpos0 = np.broadcast_to(env.data.qpos.astype(np.float32), (nworld, env.model.nq)).copy()
    qvel0 = np.broadcast_to(env.data.qvel.astype(np.float32), (nworld, env.model.nv)).copy()

    actions = wp.from_torch(torch.randn((H, nworld, env.action_dim), device = device))

    def rollout_body():
        for h in range(H):
            wp.copy(env._wd.ctrl, actions[h])
            mjw.step(env._wm, env._wd)
           

    with wp.ScopedCapture() as cap:
        rollout_body()
    graph = cap.graph

    times = []
    for _ in range(n_timed):
        wp.synchronize()
        t0 = time.perf_counter()
        env._wd.qpos.assign(qpos0)
        env._wd.qvel.assign(qvel0)
        wp.capture_launch(graph)
        wp.synchronize()
        times.append(time.perf_counter() - t0)

    print("warp sample+rollout+cost no readback", np.mean(times), "s")

    t0 = time.perf_counter()
    print("cost readback once", time.perf_counter() - t0, "s")

    return times 

if __name__ == "__main__":
    K = 512
    H = 128
    n_batches = 512

    print(f"K={K} H={H} nworld={n_batches}")
    bench_cpu(K=K, H=H, n_batches=n_batches)
    bench_warp(K=K, H=H, n_batches=n_batches)


          
