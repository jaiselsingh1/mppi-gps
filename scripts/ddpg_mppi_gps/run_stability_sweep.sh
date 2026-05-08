#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

SWEEP_ID="${SWEEP_ID:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_DIR="runs/ddpg_mppi_stability_sweep_${SWEEP_ID}"
SUMMARY="${SWEEP_DIR}/summary.jsonl"
mkdir -p "${SWEEP_DIR}"

echo "sweep_dir: ${SWEEP_DIR}"
echo "summary: ${SUMMARY}"

if tmux has-session -t ddpg-mppi-track 2>/dev/null; then
  echo "waiting for existing ddpg-mppi-track baseline to finish..."
  while tmux has-session -t ddpg-mppi-track 2>/dev/null; do
    sleep 60
  done
fi

run_exp() {
  local name="$1"
  shift
  local run_name="ddpg_mppi_sweep_${SWEEP_ID}_${name}"
  local metrics="runs/${run_name}/metrics.jsonl"

  echo
  echo "===== ${name} ====="
  echo "run_name: ${run_name}"
  echo "args: $*"

  if [[ -f "${metrics}" ]] && [[ "$(wc -l < "${metrics}")" -ge 100 ]]; then
    echo "metrics already complete; summarizing existing run"
  else
    uv run python scripts/ddpg_mppi_gps/train.py \
      --run-name "${run_name}" \
      --device cuda \
      --num-episodes 100 \
      --steps-per-episode 500 \
      --eval-every 10 \
      --eval-episodes 10 \
      --eval-steps 500 \
      --checkpoint-every 100 \
      "$@"
  fi

  jq -s -c --arg name "${name}" --arg run_name "${run_name}" '
    {
      name: $name,
      run_name: $run_name,
      episodes: length,
      last_episode: .[-1].episode,
      last10_mean_cost: (.[-10:] | map(.episode_cost) | add / length),
      last10_hit_rate: (.[-10:] | map(if .hit_success then 1 else 0 end) | add / length),
      last10_hold_rate: (.[-10:] | map(if .hold_success then 1 else 0 end) | add / length),
      evals: [ .[] | select(has("eval_mean_cost")) | {
        episode,
        eval_mean_cost,
        eval_hit_success_rate,
        eval_hold_success_rate,
        eval_mean_final_tip_dist,
        eval_mean_final_qvel_norm
      }],
      best_eval_cost: ([ .[] | select(has("eval_mean_cost")) | .eval_mean_cost ] | min),
      best_eval_hit_rate: ([ .[] | select(has("eval_hit_success_rate")) | .eval_hit_success_rate ] | max),
      final_eval: ([ .[] | select(has("eval_mean_cost")) ][-1] | {
        episode,
        eval_mean_cost,
        eval_hit_success_rate,
        eval_hold_success_rate,
        eval_mean_final_tip_dist,
        eval_mean_final_qvel_norm
      })
    }
  ' "${metrics}" | tee -a "${SUMMARY}"

  uv run python scripts/ddpg_mppi_gps/plot_sweep_results.py \
    --sweep-id "${SWEEP_ID}" \
    --include-baseline || true
}

run_exp "critic3e-4_track1e-3" \
  --critic-lr 0.0003 \
  --actor-lr 0.0001 \
  --reward-scale 1.0 \
  --lambda-policy-track 0.001 \
  --tracking-warmup-steps 1000

run_exp "reward0p1_critic1e-3_track1e-3" \
  --critic-lr 0.001 \
  --actor-lr 0.0001 \
  --reward-scale 0.1 \
  --lambda-policy-track 0.001 \
  --tracking-warmup-steps 1000

run_exp "reward0p1_critic3e-4_track1e-3" \
  --critic-lr 0.0003 \
  --actor-lr 0.0001 \
  --reward-scale 0.1 \
  --lambda-policy-track 0.001 \
  --tracking-warmup-steps 1000

run_exp "track1e-4_warm1k" \
  --critic-lr 0.0003 \
  --actor-lr 0.0001 \
  --reward-scale 0.1 \
  --lambda-policy-track 0.0001 \
  --tracking-warmup-steps 1000

run_exp "track1e-3_warm5k" \
  --critic-lr 0.0003 \
  --actor-lr 0.0001 \
  --reward-scale 0.1 \
  --lambda-policy-track 0.001 \
  --tracking-warmup-steps 5000

run_exp "track5e-3_warm5k" \
  --critic-lr 0.0003 \
  --actor-lr 0.0001 \
  --reward-scale 0.1 \
  --lambda-policy-track 0.005 \
  --tracking-warmup-steps 5000

run_exp "mppi_lam1e-3_track1e-3" \
  --critic-lr 0.0003 \
  --actor-lr 0.0001 \
  --reward-scale 0.1 \
  --lambda-policy-track 0.001 \
  --tracking-warmup-steps 5000 \
  --mppi-lam 0.001

run_exp "mppi_lam3e-3_track1e-3" \
  --critic-lr 0.0003 \
  --actor-lr 0.0001 \
  --reward-scale 0.1 \
  --lambda-policy-track 0.001 \
  --tracking-warmup-steps 5000 \
  --mppi-lam 0.003

echo
echo "sweep complete"
echo "summary: ${SUMMARY}"
uv run python scripts/ddpg_mppi_gps/plot_sweep_results.py \
  --sweep-id "${SWEEP_ID}" \
  --include-baseline || true
