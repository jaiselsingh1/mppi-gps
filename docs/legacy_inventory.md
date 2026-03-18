# Legacy MPPI-GPS Inventory

This file records the existing MPPI/GPS code found under `Desktop` and `Documents`
before starting a clean implementation in `mathforml/GPS`.

## Keep As References

### 1. Julia reference implementation

- Path: `/Users/jaiselsingh/Desktop/research_projects/guided_policy_search`
- GitHub: `git@github.com:jaiselsingh1/guided_policy_search.git`
- Why keep:
  - most complete Julia implementation
  - has explicit `mppi.jl`, `gps.jl`, `policy.jl`, and demo scripts
  - closest match to the proposed Julia direction in `mathforml`

### 2. Newer Python package rewrite

- Path: `/Users/jaiselsingh/Desktop/workbench/mppi_gps`
- GitHub: `git@github.com:jaiselsingh1/GPS.git`
- Why keep:
  - newer than `code_cookbook/GPS`
  - package structure is cleaner
  - useful reference for controller packaging, environments, and assets

## Good First Deletion Or Archive Candidates

### 1. Older Python GPS checkout

- Path: `/Users/jaiselsingh/Desktop/workbench/code_cookbook/GPS`
- GitHub: `git@github.com:jaiselsingh1/GPS.git`
- Why it is a retirement candidate:
  - same remote as `workbench/mppi_gps`
  - older local commit and flatter structure
  - likely superseded by `workbench/mppi_gps`
- Caution:
  - currently has uncommitted local changes in `PPO.py`, `pyproject.toml`, and `uv.lock`

### 2. Standalone Julia experiment inside `mujoco_test`

- Path: `/Users/jaiselsingh/Desktop/workbench/mujoco_test/src_code/mppi_gps.jl`
- Related path: `/Users/jaiselsingh/Desktop/workbench/mujoco_test/src_code/gps_setup.jl`
- Why it is a retirement candidate:
  - looks like an earlier experimental module
  - overlaps with the more complete `guided_policy_search` repo

### 3. Deprecated Julia experiment

- Path: `/Users/jaiselsingh/Documents/deprecated code/former project/mppi.jl`
- Why it is a retirement candidate:
  - isolated older MPPI experiment
  - lower value once `guided_policy_search` has been mined for ideas

### 4. Old notes

- Path: `/Users/jaiselsingh/Documents/Obsidian/main/former/mppi gps (old).md`
- Why it is a retirement candidate:
  - explicitly marked old

## Optional Retirement Candidates

### 1. UR5-only MPPI experiment

- Path: `/Users/jaiselsingh/Desktop/roam_research/research/mppi_ur5.py`
- GitHub: `git@github.com:jaiselsingh1/ur5_research.git`
- Delete or archive if:
  - you are not planning to reuse UR5-specific rollout code

## Recommendation

For a clean restart, keep only these active references for now:

1. `/Users/jaiselsingh/Desktop/coursework/mathforml/GPS`
2. `/Users/jaiselsingh/Desktop/research_projects/guided_policy_search`
3. `/Users/jaiselsingh/Desktop/workbench/mppi_gps`

Everything else can be archived first, then deleted once you confirm nothing useful remains.
