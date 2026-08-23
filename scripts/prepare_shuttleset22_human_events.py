from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from badminton.config import SimulationConfig


DEFAULT_SHUTTLESET_ROOT = (
    REPO_ROOT / "external/CoachAI-Projects/CoachAI-Challenge-IJCAI2023/ShuttleSet22/set"
)
DEFAULT_OUTPUT = REPO_ROOT / "data/human/shuttleset22_events.csv"
DEFAULT_MANIFEST = REPO_ROOT / "data/human/shuttleset22_events_manifest.json"

TYPE_TRANSLATION = {
    "發短球": "short service",
    "發長球": "long service",
    "放小球": "net shot",
    "擋小球": "return net",
    "勾球": "cross-court net shot",
    "切球": "drop",
    "過度切球": "passive drop",
    "過渡切球": "passive drop",
    "長球": "clear",
    "挑球": "lob",
    "防守回挑": "defensive return lob",
    "殺球": "smash",
    "點扣": "wrist smash",
    "平球": "drive",
    "小平球": "driven flight",
    "後場抽平球": "back-court drive",
    "推球": "push",
    "撲球": "rush",
    "防守回抽": "defensive return drive",
}

REASON_TRANSLATION = {
    "出界": "out",
    "落點判斷失誤": "misjudged",
    "掛網": "touched the net",
    "未過網": "not pass over the net",
    "對手落地致勝": "opponent ball landed",
    "犯規": "not pass over the net",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert public ShuttleSet22 stroke CSVs into the human-event schema used by plot_human_sanity_comparison.py."
    )
    parser.add_argument("--shuttleset-root", type=Path, default=DEFAULT_SHUTTLESET_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--drop-flawed-rallies", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-unknown-shots", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    if not np.isfinite(result):
        return None
    return float(result)


def _project_point(
    x: Any,
    y: Any,
    matrix: np.ndarray,
    bounds: tuple[float, float, float, float],
    config: SimulationConfig,
) -> tuple[float | None, float | None]:
    x_value = _float_or_none(x)
    y_value = _float_or_none(y)
    if x_value is None or y_value is None:
        return None, None
    point = matrix.dot(np.asarray([x_value, y_value, 1.0], dtype=float))
    if abs(float(point[2])) < 1e-9:
        return None, None
    point = point / point[2]
    min_x, max_x, min_y, max_y = bounds
    x_scale = config.court.width / max(max_x - min_x, 1e-9)
    y_scale = config.court.length / max(max_y - min_y, 1e-9)
    x_m = (float(point[0]) - 0.5 * (min_x + max_x)) * x_scale
    y_m = (float(point[1]) - 0.5 * (min_y + max_y)) * y_scale
    return x_m, y_m


def _canonicalize_hitter_side(
    points: dict[str, tuple[float | None, float | None]],
) -> dict[str, tuple[float | None, float | None]]:
    contact = points.get("contact")
    if contact is None or contact[1] is None or float(contact[1]) <= 0.0:
        return points
    flipped: dict[str, tuple[float | None, float | None]] = {}
    for key, value in points.items():
        x, y = value
        flipped[key] = (x, None if y is None else -float(y))
    return flipped


def _homography_bounds(row: dict[str, Any], matrix: np.ndarray) -> tuple[float, float, float, float]:
    points = []
    for x_key, y_key in (
        ("upleft_x", "upleft_y"),
        ("upright_x", "upright_y"),
        ("downleft_x", "downleft_y"),
        ("downright_x", "downright_y"),
    ):
        x = _float_or_none(row.get(x_key))
        y = _float_or_none(row.get(y_key))
        if x is None or y is None:
            continue
        point = matrix.dot(np.asarray([x, y, 1.0], dtype=float))
        point = point / point[2]
        points.append((float(point[0]), float(point[1])))
    if len(points) < 4:
        return 27.4, 327.6, 150.0, 810.0
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), max(xs), min(ys), max(ys)


def load_homographies(root: Path) -> dict[str, tuple[np.ndarray, tuple[float, float, float, float]]]:
    path = root / "homography.csv"
    result: dict[str, tuple[np.ndarray, tuple[float, float, float, float]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            matrix = np.asarray(ast.literal_eval(str(row["homography_matrix"])), dtype=float)
            result[str(row["video"])] = (matrix, _homography_bounds(row, matrix))
    return result


def _rows_by_rally(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("rally", "")), []).append(row)
    for rally_rows in grouped.values():
        rally_rows.sort(key=lambda item: int(float(item.get("ball_round") or 0)))
    return grouped


def convert_set_file(
    path: Path,
    *,
    video: str,
    set_number: int,
    matrix: np.ndarray,
    bounds: tuple[float, float, float, float],
    config: SimulationConfig,
    drop_flawed_rallies: bool,
    drop_unknown_shots: bool,
) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        raw_rows = list(csv.DictReader(handle))
    grouped = _rows_by_rally(raw_rows)
    converted: list[dict[str, Any]] = []
    for rally, rows in grouped.items():
        if drop_flawed_rallies and any(str(row.get("flaw") or "").strip() for row in rows):
            continue
        if drop_unknown_shots and any(str(row.get("type") or "").strip() == "未知球種" for row in rows):
            continue
        rally_id = f"{video}__set{set_number}__rally{rally}"
        rally_length = len(rows)
        for index, row in enumerate(rows):
            landing_x, landing_y = _project_point(row.get("landing_x"), row.get("landing_y"), matrix, bounds, config)
            contact_x, contact_y = _project_point(row.get("hit_x"), row.get("hit_y"), matrix, bounds, config)
            player_x, player_y = _project_point(
                row.get("player_location_x"),
                row.get("player_location_y"),
                matrix,
                bounds,
                config,
            )
            opponent_x, opponent_y = _project_point(
                row.get("opponent_location_x"),
                row.get("opponent_location_y"),
                matrix,
                bounds,
                config,
            )
            if contact_x is None or contact_y is None:
                contact_x, contact_y = player_x, player_y

            recovery_x = recovery_y = None
            if index + 1 < len(rows):
                next_row = rows[index + 1]
                recovery_x, recovery_y = _project_point(
                    next_row.get("opponent_location_x"),
                    next_row.get("opponent_location_y"),
                    matrix,
                    bounds,
                    config,
                )

            canonical = _canonicalize_hitter_side(
                {
                    "landing": (landing_x, landing_y),
                    "contact": (contact_x, contact_y),
                    "recovery": (recovery_x, recovery_y),
                    "opponent": (opponent_x, opponent_y),
                }
            )
            landing_x, landing_y = canonical["landing"]
            contact_x, contact_y = canonical["contact"]
            recovery_x, recovery_y = canonical["recovery"]
            opponent_x, opponent_y = canonical["opponent"]

            shot_type = TYPE_TRANSLATION.get(str(row.get("type") or "").strip(), str(row.get("type") or "").strip())
            terminal_reason = str(row.get("lose_reason") or row.get("win_reason") or "").strip()
            terminal_reason = REASON_TRANSLATION.get(terminal_reason, terminal_reason)
            converted.append(
                {
                    "source": "ShuttleSet22",
                    "match": video,
                    "set": set_number,
                    "rally_id": rally_id,
                    "rally": rally,
                    "ball_round": int(float(row.get("ball_round") or 0)),
                    "rally_length": rally_length,
                    "player": row.get("player"),
                    "shot_type": shot_type,
                    "raw_shot_type": row.get("type"),
                    "terminal_reason": terminal_reason,
                    "getpoint_player": row.get("getpoint_player"),
                    "landing_x": landing_x,
                    "landing_y": landing_y,
                    "contact_x": contact_x,
                    "contact_y": contact_y,
                    "recovery_x": recovery_x,
                    "recovery_y": recovery_y,
                    "opponent_x": opponent_x,
                    "opponent_y": opponent_y,
                    "landing_area": row.get("landing_area"),
                    "player_location_area": row.get("player_location_area"),
                    "opponent_location_area": row.get("opponent_location_area"),
                }
            )
    return converted


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = SimulationConfig()
    homographies = load_homographies(args.shuttleset_root)
    rows: list[dict[str, Any]] = []
    set_files = sorted(args.shuttleset_root.glob("*/set*.csv"))
    for set_path in set_files:
        video = set_path.parent.name
        if video not in homographies:
            continue
        matrix, bounds = homographies[video]
        set_digits = "".join(ch for ch in set_path.stem if ch.isdigit())
        set_number = int(set_digits or 0)
        rows.extend(
            convert_set_file(
                set_path,
                video=video,
                set_number=set_number,
                matrix=matrix,
                bounds=bounds,
                config=config,
                drop_flawed_rallies=args.drop_flawed_rallies,
                drop_unknown_shots=args.drop_unknown_shots,
            )
        )

    write_csv(args.out, rows)
    rally_ids = {str(row["rally_id"]) for row in rows}
    manifest = {
        "source": "ShuttleSet22",
        "source_root": str(args.shuttleset_root),
        "output": str(args.out),
        "set_files": len(set_files),
        "events": len(rows),
        "rallies": len(rally_ids),
        "drop_flawed_rallies": args.drop_flawed_rallies,
        "drop_unknown_shots": args.drop_unknown_shots,
        "coordinate_system": "ShuttleArena-centered meters, from ShuttleSet22 homography projection",
        "recovery_proxy": "next stroke's opponent_location_x/y, i.e. previous hitter location while receiving the reply",
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.out} ({len(rows)} events, {len(rally_ids)} rallies)")
    print(f"wrote {args.manifest_out}")


if __name__ == "__main__":
    main()
