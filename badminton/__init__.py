"""Stage-based badminton simulator with 2D court movement and 3D shuttle flight."""

import sys as _sys

from badminton.action_space import DiscreteActionConfig, DiscreteActionMapper
from badminton.agents import GreedyReceiver, RandomValidHitter, SafeHitter, StageAgent
from badminton.config import SimulationConfig
from badminton.curricula import (
    DEFAULT_DEFENSIVE_CURRICULUM_NAME,
    DEFAULT_DEFENSIVE_CURRICULUM_OPPONENT_PATH,
    DefensiveBackcourtCurriculumConfig,
    DefensiveBackcourtPhase,
    TrainingCurriculumSpec,
    available_training_curricula,
    build_training_curriculum,
)
from badminton.env import Badminton1DEnv, default_initial_state
from badminton.match import MatchConfig, MatchResult, MatchScore, RallyResult, reset_for_serve, run_match, run_rally
from badminton.obs import ObservationConfig, ObservationEncoder
from badminton.opponents import (
    GreedyDiscretePolicy,
    GreedyInterceptOpponent,
    RandomDiscretePolicy,
    RandomValidOpponent,
    SafeDiscretePolicy,
    SafeHeuristicOpponent,
    make_baseline_policy,
    make_opponent,
)
from badminton.playback import FrameSnapshot, MatchTrace, RallyTrace, StageTrace, build_match_trace, build_rally_trace, interpolate_stage
from badminton.pressure import (
    MatchShotPressure,
    ShotPressureIndex,
    ShotPressureWeights,
    evaluate_match_pressure,
    shot_pressure_from_record,
    shot_pressure_from_stage_trace,
    summarize_match_pressure,
)
from badminton.reset_sampling import ResetSamplingConfig
from badminton.reward_shaping import (
    DefensiveLiftRewardConfig,
    LoopPenaltyConfig,
    NetProximityRewardConfig,
    OpponentTravelRewardConfig,
    PressureRewardConfig,
    ReturnDepthRewardConfig,
)
from badminton.render import ScoreboardOverlay
from badminton.selfplay import CheckpointPool, FixedCheckpointOpponent, FrozenCheckpointOpponent, LiveModelOpponent, MixedCheckpointOpponent
from badminton.rl_env import BadmintonRLEnv, RLEnvConfig, RewardConfig
from badminton.shot_generators import (
    ANGLE_BIN_NAMES,
    ANGLE_BIN_NAMES_1D,
    ANGLE_BIN_NAMES_2D,
    LANDING_ZONE_COUNT_1D,
    POWER_BIN_NAMES,
    POWER_BIN_NAMES_1D,
    SHOT_NAME_ORDER,
    TacticAction1D,
    TacticAction2D,
    TacticLookup1D,
    TacticLookup2D,
    TacticRuntimeConfig,
    infer_shot_name,
    name_velocity_shot,
)
from badminton.state import ShotAction, StageRecord, StageState
from badminton.video import (
    TrainingProgressSample,
    VideoExportResult,
    export_match_video,
    export_rally_video,
    export_training_progress_video,
    render_match_frame,
    render_video_frame,
)

__all__ = [
    "Badminton1DEnv",
    "BadmintonRLEnv",
    "DiscreteActionConfig",
    "DiscreteActionMapper",
    "DEFAULT_DEFENSIVE_CURRICULUM_NAME",
    "DEFAULT_DEFENSIVE_CURRICULUM_OPPONENT_PATH",
    "DefensiveLiftRewardConfig",
    "DefensiveBackcourtCurriculumConfig",
    "DefensiveBackcourtPhase",
    "FrameSnapshot",
    "FixedCheckpointOpponent",
    "FrozenCheckpointOpponent",
    "GreedyDiscretePolicy",
    "GreedyReceiver",
    "GreedyInterceptOpponent",
    "LiveModelOpponent",
    "MatchConfig",
    "MatchResult",
    "MatchShotPressure",
    "MatchScore",
    "MatchTrace",
    "ObservationConfig",
    "ObservationEncoder",
    "CheckpointPool",
    "LoopPenaltyConfig",
    "MixedCheckpointOpponent",
    "NetProximityRewardConfig",
    "OpponentTravelRewardConfig",
    "RandomValidHitter",
    "RandomDiscretePolicy",
    "RandomValidOpponent",
    "ResetSamplingConfig",
    "PressureRewardConfig",
    "RewardConfig",
    "ReturnDepthRewardConfig",
    "RallyResult",
    "RallyTrace",
    "RLEnvConfig",
    "SafeDiscretePolicy",
    "SafeHitter",
    "SafeHeuristicOpponent",
    "ShotAction",
    "ShotPressureIndex",
    "ShotPressureWeights",
    "SHOT_NAME_ORDER",
    "SimulationConfig",
    "ScoreboardOverlay",
    "StageAgent",
    "StageRecord",
    "StageState",
    "StageTrace",
    "TacticAction1D",
    "TacticAction2D",
    "TacticLookup1D",
    "TacticLookup2D",
    "TacticRuntimeConfig",
    "TrainingCurriculumSpec",
    "TrainingProgressSample",
    "VideoExportResult",
    "ANGLE_BIN_NAMES",
    "ANGLE_BIN_NAMES_1D",
    "ANGLE_BIN_NAMES_2D",
    "LANDING_ZONE_COUNT_1D",
    "POWER_BIN_NAMES",
    "POWER_BIN_NAMES_1D",
    "available_training_curricula",
    "build_match_trace",
    "build_training_curriculum",
    "build_rally_trace",
    "default_initial_state",
    "export_match_video",
    "export_rally_video",
    "export_training_progress_video",
    "evaluate_match_pressure",
    "interpolate_stage",
    "make_baseline_policy",
    "make_opponent",
    "infer_shot_name",
    "name_velocity_shot",
    "render_match_frame",
    "render_video_frame",
    "reset_for_serve",
    "run_match",
    "run_rally",
    "shot_pressure_from_record",
    "shot_pressure_from_stage_trace",
    "summarize_match_pressure",
]


def _register_checkpoint_module_aliases() -> None:
    """Let checkpoints saved before the package rename resolve their classes."""

    legacy_root = "badminton1d"
    _sys.modules.setdefault(legacy_root, _sys.modules[__name__])
    current_prefix = f"{__name__}."
    for module_name, module in tuple(_sys.modules.items()):
        if module_name.startswith(current_prefix):
            legacy_name = f"{legacy_root}.{module_name.removeprefix(current_prefix)}"
            _sys.modules.setdefault(legacy_name, module)


_register_checkpoint_module_aliases()
