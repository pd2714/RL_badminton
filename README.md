fixpai

# Reinforcement Learning trained badminton agents

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

Default shuttle dynamics use square drag (`drag_square`, with `drag` kept as a compatibility alias). The shuttle is integrated numerically with speed-dependent drag:

```text
speed = sqrt(v_x^2 + v_y^2 + v_z^2)
dx/dt = v_x
dy/dt = v_y
dz/dt = v_z
dv_x/dt = -k_h * speed * v_x
dv_y/dt = -k_h * speed * v_y
dv_z/dt = -g - k_v * speed * v_z
```

using `k_h = horizontal_drag_coefficient`, `k_v = vertical_drag_coefficient`, and the simulator time step `drag_dt = 0.01`.

The latest RL runs in this repo use:

- `trajectory_mode=drag_square`
- `horizontal_drag_coefficient=0.2`
- `vertical_drag_coefficient=0.16`
- `intercept_count=20`
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

Velocity-oriented hitter actions now discretize:

- in 2D: horizontal angle `phi`, vertical launch angle `theta`, total speed `v`, `x_rec`, and `y_rec`
- in 1D: down-court speed `v_y`, `v_z`, and `y_rec`

The default 2D velocity action space uses `11` `phi` bins between the contact point's rays to the two net ends, `8` nonlinearly spaced `theta` bins from the straight-line net-clearance angle up to `65` degrees, `11` `speed` bins, and the center `3x3` recovery cells from a `5x5` recovery lattice. The CLI/config names are `--phi-bins`, `--theta-bins`, and `--speed-bins`.

The PPO interface remains a flat `Discrete` action over full combinations, with action ids decoded in `phi -> theta -> v -> recovery` order.

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

Discrete 2D action-space trajectory slider for the May 5 self-play run:

```bash
python3 scripts/demo_action_space_trajectory_slider.py
```

Continuous 2D action-space-style trajectory slider:

```bash
python3 scripts/demo_continuous_action_space_trajectory_slider.py
```

1D side-view trajectory slider:

```bash
python3 scripts/demo_trajectory_slider_1d.py
```

Save a slider snapshot:

```bash
python3 scripts/demo_trajectory_slider.py \
  --phi-init 90 \
  --theta-init 35 \
  --v-init 8 \
  --kh-init 0.18 \
  --kv-init 0.42 \
  --save-path outputs/demo_trajectory_launch_slider.png \
  --no-show
```

## PPO Training

Current PPO defaults in `scripts/train_ppo.py` are tuned around the drag-square trajectory model and the latest self-play protocol:

- `trajectory_mode=drag_square`
- `reaction_time=0.15`
- `player_speed=5.0`
- `movement_model=accelerated`
- `player_acceleration=8.0`
- `racket_length=1.6`
- `max_hitting_height=2.6`
- `intercept_count=20`
- `mirror_match_fraction=0.25`
- `loop_penalty=0.03` with `loop_window=4`
- `pressure_reward_weight=0.01`
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
- using `drag_square`, `reaction_time=0.15`, `player_speed=5.0`, `movement_model=accelerated`, `player_acceleration=8.0`, `racket_length=1.6`, `max_hitting_height=2.6`, and `intercept_count=20`
- capping rallies at `120` stages with `max_rally_penalty=1.0`
- using loop-penalty shaping (`0.1`, window `4`), with pressure and defensive-shot rewards off by default
- leaving `stage_penalty` and `stall_penalty` off by default unless explicitly enabled
- leaving mid-rally hitter action masking off by default; pass `--mask-mid-rally-hitter-actions` to re-enable strict masking
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
  --reaction-time 0.15 \
  --intercept-count 20
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

## Manuscript Figure Data And Evaluation Configuration

This section documents the inputs to every panel embedded in
[`6a19f5382c36b7ba5e5cf0b1/main.pdf`](6a19f5382c36b7ba5e5cf0b1/main.pdf).
It describes the figures actually referenced by `main.tex`, rather than similarly
named exploratory images elsewhere in the repository.

### Shared definitions

The main manuscript lineage is
`outputs/rl/selfplay_2d_recoverycfdefault_resp1_3m_varietypool70hist15recent10heur5newest_to6m_20260611`
(`MAIN_RUN`).  Unless a figure is explicitly marked otherwise, its simulator uses
2D players and drag-square 3D shuttle flight (`k_h=0.20`, `k_v=0.16`, `dt=0.01`),
accelerated player motion (speed `5.0 m/s`, acceleration `8.0 m/s^2`), racket
reach `1.6 m`, reaction time `0.15 s`, and an `11 x 8 x 5` shot action grid with
a `5 x 5` recovery grid.  The policy is trained on the left side with seed `17`
and eight environments.  Its recovery update uses CRA with coefficient `0.05`,
24 alternative recovery targets, and one opponent-response sample.

`Training pool` and `evaluation pool` are deliberately different things:

- The training pool is the changing opponent mixture used while optimizing a
  policy.  In the 3--6M continuation of `MAIN_RUN`, it uses variety sampling:
  70% historical anchors (linear-recency weighted), 15% recent continuation
  checkpoints, 5% newest continuation checkpoint, and 10% heuristic opponents.
- A fixed evaluation pool is frozen before the evaluation.  No policy is updated
  and no opponent is resampled from the training mixture.  A *pairwise cell*
  means one candidate policy against one fixed-pool policy for the stated number
  of rallies.  The row policy's rally-win fraction is the cell value.  Ratings
  are retrospective Bradley--Terry/Elo fits to all such cells (initial rating
  1500, scale 400, Gaussian prior standard deviation 400); they are not training
  rewards.

The run configuration is stored in `MAIN_RUN/selfplay_config.json`.  The
generation scripts named below are the reproducible route from their raw
artifacts to the raster embedded in the PDF.  The older JSON files in
`6a19f5382c36b7ba5e5cf0b1/figures/source_data/` are retained source-data
snapshots for early plot versions; where they disagree with the manuscript image,
the script and `MAIN_RUN` artifact cited here are authoritative.

### Figure `fig:overview` -- environment overview

The four subpanels are a schematic, not a statistical measurement or an RL
evaluation.

- **A:** a hand-specified court, players, one shuttle trajectory, and a recovery
  target.  The illustrative trajectory starts at `(-0.18, -2.95, 1.72)` with
  velocity `(1.50, 6.60, 4.15)` and uses drag-square dynamics with the diagram's
  `k_h=0.12`, `k_v=0.10`.
- **B:** the state and factorized action diagram.
- **C:** the rally-transition diagram.
- **D:** the self-play/checkpoint/evaluation pipeline diagram.

Source: [`scripts/create_figure1.py`](scripts/create_figure1.py), which writes
the component panels under `6a19f5382c36b7ba5e5cf0b1/figures/figure1/`; the
manuscript uses the composed `figures/generated_fig1.png`.  There is no fixed
pool, checkpoint sample, or uncertainty calculation for this figure.

### Figure `fig:competitive` -- frozen-checkpoint competitive evaluation

Both panels use the same 200-rally fixed-pool round robin.  The raw source is
`MAIN_RUN/anchor_metric_eval/cached_win_rate_matrix_200r/`:
`pair_results.csv`, `win_rate_matrix.csv`, `win_rate_matrix.json`,
`elo_standings.csv`, and `fixed_pool_eval_report.json`.  The manuscript image is
made from these data by
[`scripts/make_anchor_metric_winrate_rating_combined.py`](scripts/make_anchor_metric_winrate_rating_combined.py),
with the zero-step row and column removed.
This is the manuscript's split-vs-pure-recency competitive comparison: the
rating curve labels the compatible pure-recency prefix as `old` / pure recency
and the post-3M broader sampling continuation as `new` / pure+linear recency.
It is separate from the later CRA/no-CRA ablation panel.

- **A (matrix):** rows are the evaluated checkpoints and columns are the fixed
  opponents.  Both sets are the same 30 frozen anchors at 0.2, 0.4, ..., 6.0M
  self-play steps; therefore this is a 30-by-30 pairwise evaluation table.  Each
  off-diagonal cell is the row checkpoint's stochastic rally-win rate over 200
  rallies.  The report also contains the 0-step anchor, but the `no0` manuscript
  rendering excludes it.  Evaluation seed is `20260612`, and action selection is
  stochastic (`deterministic=false`).
- **B (rating curve):** the Elo/Bradley--Terry fit to the same directed 200-rally
  cells as panel A; it introduces no additional gameplay data.  The plotted
  rating for each checkpoint therefore summarizes its complete fixed-pool row,
  rather than a match against only the newest policy.
  Do not substitute the companion two-line source under
  `MAIN_RUN/cross_run_fixed_pool_eval_200r/` for this panel: that file uses a
  smaller 16-opponent cross-run fixed pool and is not the full main Fig2B matrix
  pool.

Some pair results are explicitly cached: the shared 0--3M prefix comes from the
compatible earlier lineage, and its cross-run cells come from
`MAIN_RUN/cross_run_fixed_pool_eval_200r/`.  The final report records 961
complete cells: 577 newly simulated, 372 cached, and 12 identity entries.  The
cache preserves the same 200-rally protocol; it is provenance reuse, not a
change of metric.

### Figure `fig:controlled-contact` -- conditional trajectory evolution

Each panel fixes the full contact state and then plots the highest-probability
valid shot from every anchor checkpoint at 0, 0.2, ..., 6.0M (31 checkpoints).
Thus each panel contains 31 trajectories/landing markers; the trajectory color
encodes checkpoint step.  These are conditional policy outputs, not trajectories
sampled from a match distribution.

Raw source: `MAIN_RUN/anchor_metric_eval/controlled_contact_grid_probe/`
`top3_expectation_evolution_probe_views/top3_expectation_evolution_samples.csv`.
The exact three scenario IDs are selected in
[`scripts/make_selected_controlled_contact_sample_trajectories_3d_combined_3m.py`](scripts/make_selected_controlled_contact_sample_trajectories_3d_combined_3m.py).

- **Left:** `frontcourt_left_low__opponent_frontcourt_left` -- a low left
  frontcourt contact, with the opponent fixed in the left frontcourt.
- **Middle:** `frontcourt_right_low__opponent_frontcourt_middle` -- a low right
  frontcourt contact, with the opponent fixed in the middle frontcourt.
- **Right:** `backcourt_left_high__opponent_midcourt_left` -- a high left
  backcourt contact, with the opponent fixed in the left midcourt.

The underlying contact-grid state file is
`controlled_contact_grid_probe_state.json`.  At the base scenarios, low contacts
have `z=0.5 m`, the high contact has `z=2.5 m`, and the opponent velocity is zero;
the panel-specific opponent positions are produced by the script's deterministic
grid expansion.  Each source row stores the chosen discrete action, validity,
landing point, pressure fields, and the checkpoint path, so the displayed curve
can be regenerated from the action and shared simulator configuration.

### Figure `fig:top-shots-backcourt` -- top-three shot modes

All six panels show the top three *valid discrete shot modes* of the 5.6M
checkpoint, not a rollout average.  The policy probabilities come from
`MAIN_RUN/anchor_metric_eval/controlled_contact_grid_probe/top_shot_3d_views/`
`top_shot_trajectories_3d_manifest.json` (stationary-opponent panels) and
`top_shot_trajectories_3d_opponent_velocity_5ms_manifest.json` (moving-opponent
panels).  Each plotted trajectory is re-simulated from its stored velocity using
the shared `MAIN_RUN` physics configuration.

- **Top-left, top-middle, top-right:**
  `backcourt_left_high__opponent_backcourt_right`,
  `..._middle`, and `..._left`.  The hitter contact is fixed at
  `(-1.627, -5.483, 2.5)` m; the opponent is stationary in the corresponding
  right-side backcourt cell.  Source/generator:
  [`scripts/make_backcourt_left_high_backcourt_opponent_top3_3d_combined_3m.py`](scripts/make_backcourt_left_high_backcourt_opponent_top3_3d_combined_3m.py).
- **Bottom-left:** `backcourt_middle_high__opponent_midcourt_middle`, with fixed
  hitter contact `(0, -5.483, 2.5)` m and a stationary opponent at midcourt.
- **Bottom-middle:** the same state with opponent lateral velocity
  `v_x=-5 m/s`.
- **Bottom-right:** the same state with opponent lateral velocity `v_x=+5 m/s`.
  The three bottom panels are generated by
  [`scripts/make_backcourt_middle_high_opponent_vx_top3_3d_combined_3m.py`](scripts/make_backcourt_middle_high_opponent_vx_top3_3d_combined_3m.py).

The six-panel manuscript composite is
`figures/backcourt_left_high_and_middle_high_top3_shot_trajectories_3d_2rows_common_legend.png`.
Its two row images originate in the output paths declared in the two scripts
above.  Probability grayscale is normalized separately within each panel to the
largest of that panel's three modes.

### Figure `fig:recovery-evolution` -- recovery under a fixed shot/response

Each panel holds the pre-contact state and outgoing shot fixed.  For every
checkpoint, it evaluates all 25 recovery-grid cells against the same one sampled
opponent response for that checkpoint/shot context, then plots the policy's top
and second recovery choices.  This is a controlled recovery probe, not a
pairwise-play result.  The general source is
`MAIN_RUN/anchor_metric_eval/`; the three state JSONs and CSVs are loaded by
[`scripts/make_recovery_contact_top_recovery_evolution_3d_combined_3m_selected.py`](scripts/make_recovery_contact_top_recovery_evolution_3d_combined_3m_selected.py).

- **Left:** `recovery_contact_grid_probe_x0_0_yneg2_k0_latest`, scenario
  `frontcourt_left_low`.  The hitter begins at `(0, -2.0, 1.5)` m and plays the
  fixed shot `(-2.921, 5.277, 3.521) m/s` toward a low left-frontcourt opponent
  contact.  The CSV has 31 checkpoints (0--6M in 0.2M increments) times 25
  recovery cells, or 775 rows for this scenario.
- **Middle:** `recovery_contact_grid_probe_x0_0_yneg6_k0_latest`, scenario
  `frontcourt_right_low`.  The hitter begins at `(0, -6.0, 1.5)` m and plays
  `(3.591, 18.087, 4.458) m/s` toward the low right-frontcourt contact.  It also
  contains 31 checkpoints times 25 cells (775 rows).
- **Right:** `backcourt_left_high_smash_recovery_comparison`, scenario
  `cross_positive_x_smash`.  The hitter begins at
  `(-1.627, -5.483, 2.5)` m and plays a fixed cross-court smash
  `(28.212, 75.276, -7.169) m/s`.  This comparison uses the seven whole-million
  checkpoints (0--6M), again scoring all 25 recovery cells per checkpoint (175
  rows).

The opponent-response count is one in all three probes, matching the CRA
configuration.  The CSV fields retain the response action/trajectory, recovery
cell probability, and critic score, making the top-two selection auditable.

### Figure `fig:ablations` -- recovery and CRA ablations

- **A (learned versus centered recovery):** source
  `MAIN_RUN/recovery_ablation_fixed_pool_learned_centered/elo_by_variant.csv`,
  `pair_results.csv`, `manifest.json`, and `recovery_ablation_report.json`;
  rendered by
  [`scripts/make_elo_evolution_recovery_ablation_combined.py`](scripts/make_elo_evolution_recovery_ablation_combined.py).
  Candidates are the main-lineage checkpoints at 0, 1, ..., 6M, each evaluated
  twice: the untouched learned policy and the same policy with only its recovery
  index replaced by the centered feasible grid cell.  The shot factors are
  unchanged.  The fixed pool is six frozen variants: learned and centered
  versions of the 2M, 4M, and 6M checkpoints.  Every candidate--pool pair uses
  200 stochastic rallies, side-balanced as 100 with the candidate on each side
  (seed `20260611`).  The Elo curve is fit across those pairwise results.
- **B (CRA versus no CRA; not the manuscript split-vs-pure competitive panel):** source
  `outputs/rl/cross_run_fixed_pool_0p4m_to_3p2m_200r_20260611/elo_ratings.csv`
  and `fixed_pool_eval_report.json`.  It compares checkpoints at 0.4, 0.8, ...,
  3.2M from two independently trained lineages: `recoverycfdefault` (CRA: 24
  alternatives, coefficient 0.05) and `norecoverycfadv` (CRA disabled: zero
  alternatives and coefficient 0).  The common fixed evaluation pool contains
  eight checkpoints from each lineage (16 total); every candidate--opponent cell
  has 200 stochastic rallies, side-balanced 100/100, with seed `20260611`.
  A single Bradley--Terry/Elo fit over the common-pool cells produces both lines.

### Figure `fig:rally-exhibition` -- qualitative rollouts

This figure is qualitative: a subpanel is a simulator snapshot within one
selected rally, not an aggregate statistic.  The source traces are from the
compatible earlier `recoverycfdefault` lineage,
`outputs/rl/selfplay_2d_recoverycfdefault_resp1_2m_heuristicbase_ent002_speed100_anchor100k_fullrec24_20260603/videos/checkpoint_matchups/`.

- **Top row:** all four stages of rally 1 from
  `step200k_vs_step3000k_seed20270611_backcourt/match_trace.json`: the 0.2M
  policy (left) loses to the frozen 3.0M policy (right).  This deterministic
  first-to-five match uses seed `20270611`; its separate
  `backcourt_match_manifest.json` records why this trace was selected.
- **Middle row:** all four stages of rally 1 from
  `step6000k_vs_step3000k/match_trace.json`: the 6.0M policy (left) wins.
- **Bottom two rows:** all eight stages of rally 9 from that same
  `step6000k_vs_step3000k/match_trace.json`: the 6.0M policy (left) wins the
  extended defensive/counter-attacking rally.

The 6.0M-versus-3.0M trace was generated as a deterministic
(`deterministic=true`) first-to-five match, seed `20260610`, with a random server
each rally and 10 fps.  `checkpoint_matchup_videos_manifest.json` records the
checkpoint identities, per-rally winner, server, and rally length.  The static
source panels are rendered from the stored `match_trace.json` files with
[`scripts/create_static_rally_exhibition.py`](scripts/create_static_rally_exhibition.py)
at a representative within-stage time (`sample_fraction=0.5`).  The final
vertically arranged manuscript composite is
`6a19f5382c36b7ba5e5cf0b1/figures/rally_exhibition_vertical_combined_tight.png`.
No evaluation pool or Elo calculation is involved in this figure.

## Example Self-Play Video

April 20, 2026 mirror-self checkpoint match from the defensive curriculum run:

[![Mirror self-play match preview](outputs/rl/selfplay_1d_dragsquare_defense_curriculum_20260420_100k/videos/mirror_self_5pt_200k_20260420/match.gif)](https://raw.githubusercontent.com/pd2714/RL_badminton/main/outputs/rl/selfplay_1d_dragsquare_defense_curriculum_20260420_100k/videos/mirror_self_5pt_200k_20260420/match.mp4)

Direct MP4: [match.mp4](https://raw.githubusercontent.com/pd2714/RL_badminton/main/outputs/rl/selfplay_1d_dragsquare_defense_curriculum_20260420_100k/videos/mirror_self_5pt_200k_20260420/match.mp4)

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
