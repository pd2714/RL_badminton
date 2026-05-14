from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton1d.action_space import DiscreteActionConfig
from badminton1d.config import ActionConfig, CourtConfig, PlayerConfig, SimulationConfig
from badminton1d.evaluation import ModelSelector, summarize_episodes
from badminton1d.playback import MatchTrace, RallyTrace, build_rally_trace
from badminton1d.rl_env import BadmintonRLEnv
from badminton1d.selfplay import CheckpointPool, FixedCheckpointOpponent, build_selfplay_env
from badminton1d.state import Side
from badminton1d.utils import ensure_directory
from badminton1d.video import VideoExportResult, export_match_video


@dataclass(frozen=True)
class AgentSpec:
    label: str
    run_dir: Path
    model_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run side-balanced checkpoint round robin and export 5-point match videos."
    )
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="final_model.zip")
    parser.add_argument("--eval-rallies-per-side", type=int, default=250)
    parser.add_argument("--video-target-score", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--rally-pause", type=float, default=0.2)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _load_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "selfplay_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing selfplay config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    value = config.get(key)
    return default if value is None else value


def build_sim_config(config: dict[str, Any]) -> SimulationConfig:
    defaults = SimulationConfig()
    return SimulationConfig(
        court=CourtConfig(mode=str(_config_value(config, "court_mode", defaults.court.mode))),
        player=PlayerConfig(
            v_max=float(_config_value(config, "player_speed", defaults.player.v_max)),
            r_reach=float(_config_value(config, "racket_length", defaults.player.r_reach)),
            z_max=float(_config_value(config, "max_hitting_height", defaults.player.z_max)),
            acceleration=float(_config_value(config, "player_acceleration", defaults.player.acceleration)),
            deceleration=_config_value(config, "player_deceleration", defaults.player.deceleration),
            movement_model=str(_config_value(config, "movement_model", defaults.player.movement_model)),
        ),
        action=ActionConfig(
            trajectory_mode=str(_config_value(config, "trajectory_mode", defaults.action.trajectory_mode)),
            drag_coefficient=float(_config_value(config, "drag_coefficient", defaults.action.drag_coefficient)),
            horizontal_drag_coefficient=float(
                _config_value(config, "horizontal_drag_coefficient", defaults.action.horizontal_drag_coefficient)
            ),
            vertical_drag_coefficient=float(
                _config_value(config, "vertical_drag_coefficient", defaults.action.vertical_drag_coefficient)
            ),
            vy_min_forward=float(_config_value(config, "shuttle_speed_min", defaults.action.vy_min_forward)),
            vy_max_forward=float(_config_value(config, "shuttle_speed_max", defaults.action.vy_max_forward)),
            recovery_x_margin=float(_config_value(config, "recovery_x_margin", defaults.action.recovery_x_margin)),
            recovery_net_margin=float(_config_value(config, "recovery_net_margin", defaults.action.recovery_net_margin)),
            recovery_back_margin=float(_config_value(config, "recovery_back_margin", defaults.action.recovery_back_margin)),
            intercept_count=int(_config_value(config, "intercept_count", defaults.action.intercept_count)),
            reaction_miss_fast_threshold=float(
                _config_value(config, "reaction_miss_fast_threshold", defaults.action.reaction_miss_fast_threshold)
            ),
            reaction_miss_fast_probability=float(
                _config_value(config, "reaction_miss_fast_probability", defaults.action.reaction_miss_fast_probability)
            ),
            reaction_miss_secondary_threshold=float(
                _config_value(
                    config,
                    "reaction_miss_secondary_threshold",
                    defaults.action.reaction_miss_secondary_threshold,
                )
            ),
            reaction_miss_secondary_probability=float(
                _config_value(
                    config,
                    "reaction_miss_secondary_probability",
                    defaults.action.reaction_miss_secondary_probability,
                )
            ),
            reaction_miss_zero_threshold=float(
                _config_value(config, "reaction_miss_zero_threshold", defaults.action.reaction_miss_zero_threshold)
            ),
        ),
    )


def build_discrete_action_config(config: dict[str, Any]) -> DiscreteActionConfig:
    defaults = DiscreteActionConfig()
    return DiscreteActionConfig(
        phi_bins=int(_config_value(config, "phi_bins", "vx_bins", defaults.phi_bins)),
        theta_bins=int(_config_value(config, "theta_bins", "vy_bins", defaults.theta_bins)),
        speed_bins=int(_config_value(config, "speed_bins", "vz_bins", defaults.speed_bins)),
        x_rec_bins=int(_config_value(config, "x_rec_bins", defaults.x_rec_bins)),
        y_rec_bins=int(_config_value(config, "y_rec_bins", defaults.y_rec_bins)),
    )


def _resolve_random_server(rng: np.random.Generator) -> Side:
    return "left" if bool(rng.integers(0, 2)) else "right"


def _next_server(
    *,
    current_server: Side,
    winner: object,
    random_server_each_rally: bool,
    rng: np.random.Generator,
) -> Side:
    if random_server_each_rally:
        return _resolve_random_server(rng)
    if winner in {"left", "right"}:
        return str(winner)  # type: ignore[return-value]
    return current_server


def build_checkpoint_env(
    *,
    train_side: Side,
    train_config: dict[str, Any],
    sim_config: SimulationConfig,
    discrete_action_config: DiscreteActionConfig,
    opponent: AgentSpec,
    seed: int,
    deterministic: bool,
    include_records_in_info: bool,
) -> BadmintonRLEnv:
    pool = CheckpointPool(
        checkpoint_dir=opponent.model_path.parent,
        pool_size=1,
        sampling_mode="newest",
        seed=seed + 17,
    )
    opponent_policy = FixedCheckpointOpponent(
        pool=pool,
        checkpoint_path=opponent.model_path,
        sim_config=sim_config,
        discrete_action_config=discrete_action_config,
        policy_type=str(_config_value(train_config, "policy_type", "velocity_oriented")),
        deterministic=deterministic,
    )
    return build_selfplay_env(
        train_side=train_side,
        mirror_train_side=False,
        mirror_match_fraction=0.0,
        initial_server="random",
        random_service_x=bool(_config_value(train_config, "random_service_x", True)),
        sim_config=sim_config,
        train_reaction_time=float(_config_value(train_config, "reaction_time", 0.15)),
        opponent_reaction_time=float(_config_value(train_config, "opponent_reaction_time", 0.15)),
        max_stages_per_rally=int(_config_value(train_config, "max_stages_per_rally", 100)),
        policy_type=str(_config_value(train_config, "policy_type", "velocity_oriented")),
        seed=seed,
        discrete_action_config=discrete_action_config,
        opponent=opponent_policy,
        include_records_in_info=include_records_in_info,
    )


def rollout_rally(
    env: BadmintonRLEnv,
    selector: ModelSelector,
    *,
    seed: int,
    server: Side,
) -> dict[str, object]:
    observation, info = env.reset(seed=seed, options={"server": server, "train_side": env.rl_config.train_side})
    terminated = False
    truncated = False
    total_reward = 0.0

    while not terminated and not truncated:
        action = selector.choose_action(observation, env.current_decision_context())
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

    metrics = info["badminton_metrics"]
    return {
        "reward": total_reward,
        "winner": metrics["winner"],
        "rally_won": metrics["rally_won"],
        "rally_length": metrics["rally_length"],
        "invalid_action_rate": metrics["invalid_action_rate"],
        "metrics": metrics,
        "records": list(env.records),
        "config": env.config,
        "server": info["server"],
        "train_side": info["train_side"],
        "opponent_label": info.get("opponent_label"),
        "truncated": truncated,
    }


def evaluate_ordered_matchup(
    *,
    agent: AgentSpec,
    opponent: AgentSpec,
    train_side: Side,
    model: PPO,
    train_config: dict[str, Any],
    sim_config: SimulationConfig,
    discrete_action_config: DiscreteActionConfig,
    episodes: int,
    seed: int,
    deterministic: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    env = build_checkpoint_env(
        train_side=train_side,
        train_config=train_config,
        sim_config=sim_config,
        discrete_action_config=discrete_action_config,
        opponent=opponent,
        seed=seed,
        deterministic=deterministic,
        include_records_in_info=False,
    )
    selector = ModelSelector(model=model, deterministic=deterministic)
    rng = np.random.default_rng(seed)
    results = [
        rollout_rally(env, selector, seed=seed + episode, server=_resolve_random_server(rng))
        for episode in range(episodes)
    ]
    summary = summarize_episodes(results)
    summary["name"] = f"{agent.label}_as_{train_side}_vs_{opponent.label}"
    summary["agent"] = agent.label
    summary["opponent"] = opponent.label
    summary["train_side"] = train_side
    return summary, results


def _rally_trace_from_result(
    result: dict[str, object],
    config: SimulationConfig,
    *,
    rally_number: int,
    score_before_left: int,
    score_before_right: int,
    score_after_left: int,
    score_after_right: int,
    rally_pause: float,
) -> RallyTrace:
    records = result["records"]
    assert isinstance(records, list)
    trace = build_rally_trace(records, config)
    return RallyTrace(
        stages=trace.stages,
        rally_done=trace.rally_done,
        winner=trace.winner,
        total_playback_time=trace.total_playback_time,
        rally_number=rally_number,
        server=str(result["server"]),  # type: ignore[arg-type]
        score_before_left=score_before_left,
        score_before_right=score_before_right,
        score_after_left=score_after_left,
        score_after_right=score_after_right,
        pause_duration=rally_pause,
        match_winner=None,
    )


def build_match_trace(traces: list[RallyTrace], score_left: int, score_right: int) -> MatchTrace:
    winner: Side | None
    if score_left == score_right:
        winner = None
    else:
        winner = "left" if score_left > score_right else "right"
    total_playback_time = sum(trace.total_playback_time + trace.pause_duration for trace in traces)
    if traces and winner is not None:
        last = traces[-1]
        traces[-1] = RallyTrace(
            stages=last.stages,
            rally_done=last.rally_done,
            winner=last.winner,
            total_playback_time=last.total_playback_time,
            rally_number=last.rally_number,
            server=last.server,
            score_before_left=last.score_before_left,
            score_before_right=last.score_before_right,
            score_after_left=last.score_after_left,
            score_after_right=last.score_after_right,
            pause_duration=last.pause_duration,
            match_winner=winner,
        )
    return MatchTrace(
        rallies=traces,
        target_score=max(score_left, score_right),
        score_left=score_left,
        score_right=score_right,
        winner=winner,
        total_playback_time=total_playback_time,
    )


def ensure_mp4(video_result: VideoExportResult, fps: int) -> Path | None:
    if video_result.mp4_path is not None and video_result.mp4_path.exists():
        return video_result.mp4_path
    if not video_result.frame_paths:
        return None
    frame_pattern = video_result.frame_paths[0].parent / "frame_%05d.png"
    output_path = video_result.trace_path.parent / "match.mp4"
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return None
    return output_path if output_path.exists() else None


def export_pair_video(
    *,
    left_agent: AgentSpec,
    right_agent: AgentSpec,
    left_model: PPO,
    left_config: dict[str, Any],
    sim_config: SimulationConfig,
    discrete_action_config: DiscreteActionConfig,
    target_score: int,
    output_dir: Path,
    seed: int,
    fps: int,
    rally_pause: float,
    deterministic: bool,
) -> dict[str, object]:
    ensure_directory(output_dir)
    env = build_checkpoint_env(
        train_side="left",
        train_config=left_config,
        sim_config=sim_config,
        discrete_action_config=discrete_action_config,
        opponent=right_agent,
        seed=seed,
        deterministic=deterministic,
        include_records_in_info=True,
    )
    selector = ModelSelector(model=left_model, deterministic=deterministic)
    rng = np.random.default_rng(seed)
    current_server = _resolve_random_server(rng)
    score_left = 0
    score_right = 0
    traces: list[RallyTrace] = []
    summaries: list[dict[str, object]] = []

    for rally_number in range(1, 10_000):
        result = rollout_rally(env, selector, seed=seed + rally_number, server=current_server)
        winner = result["winner"]
        next_score_left = score_left + (1 if winner == "left" else 0)
        next_score_right = score_right + (1 if winner == "right" else 0)
        traces.append(
            _rally_trace_from_result(
                result,
                sim_config,
                rally_number=rally_number,
                score_before_left=score_left,
                score_before_right=score_right,
                score_after_left=next_score_left,
                score_after_right=next_score_right,
                rally_pause=rally_pause,
            )
        )
        summaries.append(
            {
                "rally_number": rally_number,
                "server": current_server,
                "winner": winner,
                "score_after_left": next_score_left,
                "score_after_right": next_score_right,
                "rally_length": result["rally_length"],
                "invalid_action_rate": result["invalid_action_rate"],
            }
        )
        score_left = next_score_left
        score_right = next_score_right
        if max(score_left, score_right) >= target_score:
            break
        current_server = _next_server(
            current_server=current_server,
            winner=winner,
            random_server_each_rally=True,
            rng=rng,
        )

    match_trace = build_match_trace(traces, score_left, score_right)
    video_result = export_match_video(match_trace, sim_config, output_dir, fps=fps, write_mp4=False)
    mp4_path = ensure_mp4(video_result, fps)
    return {
        "left_agent": left_agent.label,
        "right_agent": right_agent.label,
        "target_score": target_score,
        "score_left": score_left,
        "score_right": score_right,
        "winner": left_agent.label if score_left > score_right else right_agent.label,
        "random_server_each_rally": True,
        "gif_path": str(video_result.gif_path),
        "mp4_path": None if mp4_path is None else str(mp4_path),
        "trace_path": str(video_result.trace_path),
        "rallies": summaries,
    }


def combine_pair_summary(
    *,
    agent_a: AgentSpec,
    agent_b: AgentSpec,
    a_left_summary: dict[str, object],
    a_right_summary: dict[str, object],
) -> dict[str, object]:
    total = int(a_left_summary["episodes"]) + int(a_right_summary["episodes"])
    wins = (
        float(a_left_summary["win_rate"]) * int(a_left_summary["episodes"])
        + float(a_right_summary["win_rate"]) * int(a_right_summary["episodes"])
    )
    return {
        "pair": f"{agent_a.label}_vs_{agent_b.label}",
        "agent_a": agent_a.label,
        "agent_b": agent_b.label,
        "episodes": total,
        "agent_a_win_rate": wins / max(total, 1),
        "agent_b_win_rate": 1.0 - wins / max(total, 1),
        "agent_a_as_left": a_left_summary,
        "agent_a_as_right": a_right_summary,
    }


def make_agents(run_dirs: list[Path], checkpoint_name: str) -> list[AgentSpec]:
    agents: list[AgentSpec] = []
    seen: set[str] = set()
    for run_dir in run_dirs:
        label = run_dir.name
        if label in seen:
            raise ValueError(f"Duplicate run label: {label}")
        model_path = run_dir / checkpoint_name
        if not model_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {model_path}")
        agents.append(AgentSpec(label=label, run_dir=run_dir, model_path=model_path))
        seen.add(label)
    if len(agents) < 2:
        raise ValueError("At least two run directories are required.")
    return agents


def main() -> None:
    args = parse_args()
    _require_positive("--eval-rallies-per-side", args.eval_rallies_per_side)
    _require_positive("--video-target-score", args.video_target_score)
    _require_positive("--fps", args.fps)
    if args.rally_pause < 0.0:
        raise ValueError("--rally-pause must be non-negative")

    ensure_directory(args.output_dir)
    agents = make_agents(args.run_dirs, args.checkpoint_name)
    configs = {agent.label: _load_config(agent.run_dir) for agent in agents}
    base_config = configs[agents[0].label]
    sim_config = build_sim_config(base_config)
    discrete_action_config = build_discrete_action_config(base_config)
    models = {agent.label: PPO.load(agent.model_path) for agent in agents}

    pair_summaries: list[dict[str, object]] = []
    video_summaries: list[dict[str, object]] = []

    for pair_index, (i, j) in enumerate((i, j) for i in range(len(agents)) for j in range(i + 1, len(agents))):
        agent_a = agents[i]
        agent_b = agents[j]
        seed_base = args.seed + pair_index * 100_000
        a_left_summary, _ = evaluate_ordered_matchup(
            agent=agent_a,
            opponent=agent_b,
            train_side="left",
            model=models[agent_a.label],
            train_config=configs[agent_a.label],
            sim_config=sim_config,
            discrete_action_config=discrete_action_config,
            episodes=args.eval_rallies_per_side,
            seed=seed_base + 1_000,
            deterministic=args.deterministic,
        )
        a_right_summary, _ = evaluate_ordered_matchup(
            agent=agent_a,
            opponent=agent_b,
            train_side="right",
            model=models[agent_a.label],
            train_config=configs[agent_a.label],
            sim_config=sim_config,
            discrete_action_config=discrete_action_config,
            episodes=args.eval_rallies_per_side,
            seed=seed_base + 2_000,
            deterministic=args.deterministic,
        )
        pair_summaries.append(
            combine_pair_summary(
                agent_a=agent_a,
                agent_b=agent_b,
                a_left_summary=a_left_summary,
                a_right_summary=a_right_summary,
            )
        )
        video_summaries.append(
            export_pair_video(
                left_agent=agent_a,
                right_agent=agent_b,
                left_model=models[agent_a.label],
                left_config=configs[agent_a.label],
                sim_config=sim_config,
                discrete_action_config=discrete_action_config,
                target_score=args.video_target_score,
                output_dir=args.output_dir / "videos" / f"{agent_a.label}__vs__{agent_b.label}",
                seed=seed_base + 3_000,
                fps=args.fps,
                rally_pause=args.rally_pause,
                deterministic=args.deterministic,
            )
        )

    standings: dict[str, dict[str, float]] = {
        agent.label: {"pair_wins": 0.0, "pair_count": 0.0, "rally_win_rate_sum": 0.0}
        for agent in agents
    }
    for summary in pair_summaries:
        agent_a = str(summary["agent_a"])
        agent_b = str(summary["agent_b"])
        a_wr = float(summary["agent_a_win_rate"])
        b_wr = float(summary["agent_b_win_rate"])
        standings[agent_a]["pair_count"] += 1.0
        standings[agent_b]["pair_count"] += 1.0
        standings[agent_a]["rally_win_rate_sum"] += a_wr
        standings[agent_b]["rally_win_rate_sum"] += b_wr
        if a_wr > b_wr:
            standings[agent_a]["pair_wins"] += 1.0
        elif b_wr > a_wr:
            standings[agent_b]["pair_wins"] += 1.0
        else:
            standings[agent_a]["pair_wins"] += 0.5
            standings[agent_b]["pair_wins"] += 0.5

    standings_payload = [
        {
            "agent": label,
            "pair_wins": values["pair_wins"],
            "pair_count": values["pair_count"],
            "mean_pairwise_rally_win_rate": values["rally_win_rate_sum"] / max(values["pair_count"], 1.0),
        }
        for label, values in standings.items()
    ]
    standings_payload.sort(
        key=lambda item: (float(item["pair_wins"]), float(item["mean_pairwise_rally_win_rate"])),
        reverse=True,
    )

    report = {
        "agents": [
            {"label": agent.label, "run_dir": str(agent.run_dir), "model_path": str(agent.model_path)}
            for agent in agents
        ],
        "checkpoint_name": args.checkpoint_name,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "random_server_each_rally": True,
        "eval_rallies_per_side": args.eval_rallies_per_side,
        "eval_rallies_per_pair": args.eval_rallies_per_side * 2,
        "video_target_score": args.video_target_score,
        "pair_summaries": pair_summaries,
        "standings": standings_payload,
        "videos": video_summaries,
    }
    report_path = args.output_dir / "round_robin_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"report: {report_path}")
    for item in standings_payload:
        print(
            f"{item['agent']}: pair_wins={item['pair_wins']:.1f}/{item['pair_count']:.0f} "
            f"mean_pairwise_wr={item['mean_pairwise_rally_win_rate']:.3f}"
        )
    for summary in pair_summaries:
        print(
            f"{summary['pair']}: {summary['agent_a']} wr={summary['agent_a_win_rate']:.3f}, "
            f"{summary['agent_b']} wr={summary['agent_b_win_rate']:.3f}"
        )
    for video in video_summaries:
        print(f"video {video['left_agent']} vs {video['right_agent']}: {video['mp4_path'] or video['gif_path']}")


if __name__ == "__main__":
    main()
