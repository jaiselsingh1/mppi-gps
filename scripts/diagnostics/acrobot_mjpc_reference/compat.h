// MuJoCo 3.6 added an optional surface-normal output to mj_ray. Upstream MJPC
// still calls the older form in its Ground helper (unused by Acrobot).
#ifndef ACROBOT_MJPC_REFERENCE_COMPAT_H_
#define ACROBOT_MJPC_REFERENCE_COMPAT_H_
#include <mujoco/mujoco.h>

inline mjtNum mj_ray(const mjModel* model, const mjData* data,
                     const mjtNum point[3], const mjtNum direction[3],
                     const mjtByte* geom_group, mjtByte include_static,
                     int exclude_body, int* geom_id) {
  return mj_ray(model, data, point, direction, geom_group, include_static,
                exclude_body, geom_id, nullptr);
}
#endif
