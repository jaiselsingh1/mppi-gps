"""Gait measures for logging (not used by the method): a stride section and the walks' spread."""

from __future__ import annotations

import numpy as np

THIGH = 4  # state index of Walker's right thigh angle (time, x, z, pitch, thigh, ...)
THIGH_VEL = 13  # its velocity (qvel starts at index 10)
SKIP = 188  # controls ignored at the start (1.5 s)
GAP = 50  # controls between accepted crossings (0.4 s)
MIN_STRIDES = 5
MIN_POINTS = 3


class Section:
    def __init__(self, walks, thigh=THIGH, thigh_vel=THIGH_VEL):
        """walks: list of (T, n) walking state arrays that fix level, speed and scale;
        thigh, thigh_vel: state indices of the leg joint the section watches (Walker's right thigh
        by default; Hopper's thigh is 4 and 10)."""
        self.thigh, self.thigh_vel = thigh, thigh_vel
        pooled = np.concatenate([w[SKIP:] for w in walks])
        self.sd = np.concatenate(walks)[:, 2:].std(0) + 1e-6
        self.level = float(np.median(pooled[:, thigh]))
        self.vmin = 0.5 * float(np.percentile(pooled[:, thigh_vel], 90))

    def points(self, traj):
        """Section states and crossing times (controls) along one run."""
        a, v = traj[:, self.thigh], traj[:, self.thigh_vel]
        out, times, last = [], [], -(10**9)
        for t in range(SKIP, len(traj) - 1):
            if a[t] < self.level <= a[t + 1] and v[t] > self.vmin and t - last >= GAP:
                w = (self.level - a[t]) / (a[t + 1] - a[t])
                out.append(((1 - w) * traj[t] + w * traj[t + 1])[2:] / self.sd)
                times.append(t + w)
                last = t
        return np.array(out), np.array(times)

    def regularity(self, runs):
        """Over runs with >= 5 strides: change (last three strides), spread, stride period."""
        points, periods = [], []
        for traj in runs:
            p, times = self.points(traj)
            if len(p) >= MIN_STRIDES:
                points.append(p)
                periods.append(np.diff(times) * 0.008)
        if len(points) < 3:
            return None
        last = np.array([p[-1] for p in points])
        period = np.concatenate(periods)
        return dict(
            runs=len(points),
            change=float(np.mean([np.sqrt(((p[-3:] - p[-4:-1]) ** 2).mean()) for p in points])),
            change_first=float(np.mean([np.sqrt(((p[1:4] - p[:3]) ** 2).mean()) for p in points])),
            spread=float(np.sqrt(((last - last.mean(0)) ** 2).mean())),
            period_s=float(period.mean()),
            period_sd_s=float(period.std()),
        )


def rms(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def measure(section, walks):
    """The walks' spread at the section: W within a walk, B between walks' means, mode count."""
    means, within, periods = [], [], []
    for w in walks:
        p, times = section.points(w)
        if len(p) < MIN_POINTS:
            continue
        m = p.mean(0)
        means.append(m)
        within.append(float(np.mean([rms(x, m) for x in p])))
        periods.append(float(np.diff(times).mean() * 0.008))
    if len(means) < 2:
        return dict(walks=len(means))
    means = np.array(means)
    W = float(np.mean(within))
    B = float(np.sqrt(((means - means.mean(0)) ** 2).mean()))
    labels = list(range(len(means)))  # single linkage: join walks within 2 W
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            if rms(means[i], means[j]) <= 2 * W:
                a, b = labels[i], labels[j]
                labels = [a if x == b else x for x in labels]
    return dict(
        walks=len(means),
        W=W,
        B=B,
        R=B / W,
        modes=len(set(labels)),
        periods_s=[round(x, 3) for x in periods],
    )
