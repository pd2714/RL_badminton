# Badminton 2D Court Simulator

This repo keeps the original stage-based badminton simulator structure, but the simulator itself is now a true 2D court model built in the existing `badminton1d` package.

The package name is historical. The simulator is now:

- 2D on the ground plane with player positions `(x, y)`
- 3D for the shuttle with `(x, y, z)`
- still stage-based: one stage equals one hit
- still compatible with match logic, rendering, playback, Gym RL, PPO training, and self-play

## Coordinate System

Singles court coordinates use:

- `x`: left-right across court width
- `y`: along court length
- `z`: shuttle height

Conventions:

- left player side: `y < 0`
- right player side: `y > 0`
- net plane: `y = 0`

Default singles dimensions:

- court length: `13.4 m`
- court width: `5.18 m`
- net height: `1.55 m`

## State And Actions

Stage state contains:

- `x_left, y_left`
- `x_right, y_right`
- `current_hitter`
- shuttle hit point `x0, y0, z0`
- rally flags and stage index

Hitter action is now launch-based:

- `v_x`
- `v_y`
- `v_z`
- `x_rec`
- `y_rec`

Receiver action is still an intercept choice:

- choose one sampled intercept index along the incoming trajectory

## Shuttle Dynamics

Default mode is ballistic:

```text
x(t) = x0 + v_x t
y(t) = y0 + v_y t
z(t) = z0 + v_z t - 0.5 g t^2
```

Optional square-drag mode is supported via `drag_square` (with `drag` kept as a compatibility alias).

The latest RL runs in this repo use:

- `trajectory_mode=drag_square`
- `horizontal_drag_coefficient=0.2`
- `vertical_drag_coefficient=0.16`
- `intercept_count=50`
- `court-mode 1d` for the April 2026 self-play and defense-curriculum checkpoints

Clean helpers are available in `badminton1d/trajectory.py` and `badminton1d/dynamics.py` for:

- landing time
- landing point
- net crossing
- trajectory simulation
- hit validity
- candidate intercept sampling

## Valid Hit Rule

A hitter action is valid only if:

1. the shuttle crosses the net from the hitter side
2. it clears the net by `net_height + net_clearance_margin`
3. it lands on the opponent side
4. it lands in singles bounds

Serve stages keep the previous simplified serve logic:

- server starts near the center of the serving side
- receiver starts near the center of the opposite side
- serve must also land beyond the service line on the receiver side

## Intercept Rule

Receiver intercept feasibility is now 2D.

A sampled intercept at time `t` is feasible if:

1. ground distance is reachable:

```text
distance((x_r, y_r), (x(t), y(t))) <= v_max * available_time + r_reach
```

2. shuttle height is playable:

```text
z_min <= z(t) <= z_max
```

If no feasible intercept exists, the hitter wins the rally.

In the RL environment, an invalid receiver discrete action is also treated as an immediate loss when `invalid_receiver_choice_loses=True` (the current default).

When an intercept is chosen:

- the receiver moves to the intercept ground point
- the hitter partially recovers toward `(x_rec, y_rec)`
- the intercept point becomes the next stage hit state
- side to hit swaps

## Rendering And Video

Rendering was upgraded instead of replaced.

Current renderer:

- top-view court
- court boundaries and net line
- player positions
- shuttle ground path
- landing point
- intercept point
- recovery target
- score / server / stage overlays

Continuous rally and match video export still works:

- 2D player motion is animated
- shuttle ground projection is animated with marker size reflecting height
- GIF and MP4 export remain supported

## RL Interface

The RL stack is still intact:

- `badminton1d/rl_env.py`
- `badminton1d/obs.py`
- `badminton1d/action_space.py`
- `badminton1d/selfplay.py`

Observation now includes:

- both player 2D positions
- current shuttle hit point `(x0, y0, z0)`
- side-to-act / role flags
- score and stage progress
- pending hitter action features for receiver turns
- feasible intercept mask

Discrete hitter actions now discretize:

- `v_x`
- `v_y`
- `v_z`
- `x_rec`
- `y_rec`

## Tests

Coverage now includes:

- 2D net-crossing validity
- in-bounds landing checks
- 2D intercept feasibility
- 2D partial recovery
- stage transition correctness
- serve reset positions
- rendering and video smoke tests
- RL env observation/action compatibility

Run everything with:

```bash
python3 -m pytest -q
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Demos

Heuristic rally with per-stage renders:

```bash
python3 scripts/demo_rally.py
```

Continuous rally video:

```bash
python3 scripts/demo_video.py
```

Trajectory slider:

```bash
python3 scripts/demo_trajectory_slider.py
```

1D side-view trajectory slider:

```bash
python3 scripts/demo_trajectory_slider_1d.py
```

Save a slider snapshot:

```bash
python3 scripts/demo_trajectory_slider.py \
  --vx-init 0.8 \
  --vy-init 5.5 \
  --vz-init 5.0 \
  --kh-init 0.18 \
  --kv-init 0.42 \
  --save-path outputs/demo_trajectory_launch_slider.png \
  --no-show
```

## PPO Training

Current PPO defaults in `scripts/train_ppo.py` are tuned around the drag-square trajectory model and the latest self-play protocol:

- `trajectory_mode=drag_square`
- `reaction_time=0.3`
- `player_speed=2.6`
- `intercept_count=50`
- `mirror_match_fraction=0.25`
- `loop_penalty=0.03` with `loop_window=4`
- `pressure_reward_weight=0.05`
- `max_rally_stages=120`
- `n_envs=8`
- `initial_server=random`
- `stage_penalty=0.0` and `stall_penalty=0.0` by default

Train PPO against the heuristic opponent:

```bash
python3 scripts/train_ppo.py \
  --train-side left \
  --opponent safe \
  --total-timesteps 200000 \
  --n-envs 4
```

Evaluate a checkpoint:

```bash
python3 scripts/eval_ppo.py \
  --model-path outputs/rl/ppo_run/final_model.zip
```

Self-play training:

```bash
python3 scripts/train_selfplay.py \
  --base-checkpoint-path outputs/rl/ppo_run/final_model.zip
```

The self-play trainer in `scripts/train_selfplay.py` now defaults to:

- drawing opponents from a checkpoint pool with `recency` sampling
- using `checkpoint_recency_power=3.0`
- weighting recent vs older checkpoints `0.9 / 0.1`
- mixing in the safe heuristic opponent with probability `0.05`
- saving checkpoint-pool snapshots every `2000` timesteps
- evaluating `current_vs_newest_checkpoint` every `5000` timesteps by default
- keeping mirror-side training active for `25%` of episodes unless overridden with `--mirror-match-fraction`
- using `drag_square`, `reaction_time=0.3`, `player_speed=2.6`, and `intercept_count=50`
- capping rallies at `120` stages with `max_rally_penalty=1.0`
- using loop-penalty shaping (`0.03`, window `4`) plus pressure reward (`0.05`)
- leaving `stage_penalty` and `stall_penalty` off by default unless explicitly enabled
- not recording training-progress videos unless `--progress-video-freq` is enabled
- using the side-view 1D visualization when `--court-mode 1d` is selected

Retrain for the 1D court:

```bash
python3 scripts/train_selfplay.py \
  --base-checkpoint-path outputs/rl/ppo_1d_drag02_slow26_rt03_lp003_pr005_20260414/best_model/best_model.zip \
  --output-dir outputs/rl/selfplay_1d_newest_mirror_20260414 \
  --court-mode 1d
```

Defensive curriculum against the April 19 attack model:

```bash
python3 scripts/train_selfplay.py \
  --base-checkpoint-path outputs/rl/selfplay_1d_dragsquare_ps35_rt03_ic50_recency_masked_rm03_20260419_20k_100ktotal_attack_best/best_model.zip \
  --curriculum defensive-backcourt-attack-best \
  --output-dir outputs/rl/selfplay_1d_dragsquare_defense_curriculum_20260420_100k \
  --court-mode 1d \
  --trajectory-mode drag_square \
  --reaction-time 0.3 \
  --intercept-count 50
```

That preset:

- fixes the opponent to `outputs/rl/selfplay_1d_dragsquare_ps35_rt03_ic50_recency_masked_rm03_20260419_20k_100ktotal_attack_best/best_model.zip`
- starts rallies from a non-serve back-court attack state with the opponent as the hitter
- samples the opponent's opening attack stochastically while keeping its defensive responses deterministic
- widens the attacker position, defender displacement, and contact height over curriculum phases

The curriculum phases currently used are:

- `stabilize_center_lane` from episode `0`
- `expand_attack_angles` from episode `1500`
- `full_backcourt_pressure` from episode `4500`

Rules used by that defensive curriculum:

- every episode starts from a randomized mid-rally back-court attack instead of a normal serve
- the opponent is forced to be the initial hitter and initial server
- opening attack locations and contact heights are sampled from phase-specific ranges
- the opponent's hitter policy stays stochastic, but its defensive receiver choice is deterministic
- the defender must still satisfy the normal intercept-feasibility, net-clearance, and in-bounds rules

## Example Self-Play Video

April 20, 2026 mirror-self checkpoint match from the defensive curriculum run:

[![Mirror self-play match preview](outputs/rl/selfplay_1d_dragsquare_defense_curriculum_20260420_100k/videos/mirror_self_5pt_200k_20260420/match.gif)](outputs/rl/selfplay_1d_dragsquare_defense_curriculum_20260420_100k/videos/mirror_self_5pt_200k_20260420/match.mp4)

Direct MP4: [outputs/rl/selfplay_1d_dragsquare_defense_curriculum_20260420_100k/videos/mirror_self_5pt_200k_20260420/match.mp4](outputs/rl/selfplay_1d_dragsquare_defense_curriculum_20260420_100k/videos/mirror_self_5pt_200k_20260420/match.mp4)

Export rally sequence video:

```bash
python3 scripts/export_rally_sequence_video.py \
  --model-path outputs/rl/ppo_run/final_model.zip \
  --output-dir outputs/rollout_videos/demo_sequence
```

Render `current vs newest checkpoint`:

```bash
python3 scripts/export_rally_sequence_video.py \
  --model-path outputs/rl/selfplay_1d_newest_mirror_20260414/final_model.zip \
  --checkpoint-pool-dir outputs/rl/selfplay_1d_newest_mirror_20260414/checkpoint_pool \
  --opponent newest-checkpoint \
  --output-dir outputs/rollout_videos/current_vs_newest
```

Render `current vs mirror self`:

```bash
python3 scripts/export_rally_sequence_video.py \
  --model-path outputs/rl/selfplay_1d_newest_mirror_20260414/final_model.zip \
  --opponent mirror-self \
  --output-dir outputs/rollout_videos/current_vs_mirror_self
```
