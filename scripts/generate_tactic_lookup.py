from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton.config import ActionConfig, CourtConfig, PlayerConfig, SimulationConfig
from badminton.shot_generators import TacticLookup1D, TacticLookup2D, TacticRuntimeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cached tactic-oriented lookup tables.")
    parser.add_argument("--court-mode", choices=("1d", "2d", "both"), default="both")
    parser.add_argument("--lookup-table-dir", type=Path, default=Path("lookup_tables"))
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--trajectory-mode", choices=("ballistic", "drag", "drag_square"), default="drag_square")
    parser.add_argument("--player-speed", type=float, default=SimulationConfig().player.v_max)
    parser.add_argument("--racket-length", type=float, default=SimulationConfig().player.r_reach)
    parser.add_argument("--max-hitting-height", type=float, default=SimulationConfig().player.z_max)
    parser.add_argument("--movement-model", choices=("constant_velocity", "accelerated"), default=SimulationConfig().player.movement_model)
    parser.add_argument("--player-acceleration", type=float, default=SimulationConfig().player.acceleration)
    parser.add_argument("--player-deceleration", type=float, default=None)
    parser.add_argument("--drag-coefficient", type=float, default=0.2)
    parser.add_argument("--horizontal-drag-coefficient", type=float, default=0.2)
    parser.add_argument("--vertical-drag-coefficient", type=float, default=0.16)
    parser.add_argument("--shuttle-speed-min", type=float, default=0.1)
    parser.add_argument("--shuttle-speed-max", type=float, default=SimulationConfig().action.vy_max_forward)
    parser.add_argument("--intercept-count", type=int, default=20)
    return parser.parse_args()


def build_config(court_mode: str, args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        court=CourtConfig(mode=court_mode),
        player=PlayerConfig(
            v_max=args.player_speed,
            r_reach=args.racket_length,
            z_max=args.max_hitting_height,
            acceleration=args.player_acceleration,
            deceleration=args.player_deceleration,
            movement_model=args.movement_model,
        ),
        action=ActionConfig(
            trajectory_mode=args.trajectory_mode,
            drag_coefficient=args.drag_coefficient,
            horizontal_drag_coefficient=args.horizontal_drag_coefficient,
            vertical_drag_coefficient=args.vertical_drag_coefficient,
            vy_min_forward=args.shuttle_speed_min,
            vy_max_forward=args.shuttle_speed_max,
            intercept_count=args.intercept_count,
        ),
    )


def main() -> None:
    args = parse_args()
    runtime = TacticRuntimeConfig(
        regenerate_lookup_table=args.regenerate,
        lookup_dir=args.lookup_table_dir,
    )
    modes = ("1d", "2d") if args.court_mode == "both" else (args.court_mode,)

    for mode in modes:
        config = build_config(mode, args)
        if mode == "1d":
            lookup = TacticLookup1D(config, runtime)
        else:
            lookup = TacticLookup2D(config, runtime)
        lookup.ensure_loaded()
        summary = lookup.summary()
        print(
            f"{mode} lookup ready: "
            f"path={summary['lookup_path']} "
            f"valid_fraction={summary['valid_fraction']:.3f} "
            f"fallback_fraction={summary['fallback_fraction']:.3f}"
        )


if __name__ == "__main__":
    main()
