"""Stage-based badminton simulator with 2D court movement and 3D shuttle flight."""

from badminton1d.action_space import DiscreteActionConfig, DiscreteActionMapper
from badminton1d.agents import GreedyReceiver, RandomValidHitter, SafeHitter, StageAgent
from badminton1d.config import SimulationConfig
from badminton1d.curricula import (
    DEFAULT_DEFENSIVE_CURRICULUM_NAME,
    DEFAULT_DEFENSIVE_CURRICULUM_OPPONENT_PATH,
    DefensiveBackcourtCurriculumConfig,
    DefensiveBackcourtPhase,
    TrainingCurriculumSpec,
    available_training_curricula,
    build_training_curriculum,
)
from badminton1d.env import Badminton1DEnv, default_initial_state
from badminton1d.match import MatchConfig, MatchResult, MatchScore, RallyResult, reset_for_serve, run_match, run_rally
from badminton1d.obs import ObservationConfig, ObservationEncoder
from badminton1d.opponents import (
    GreedyDiscretePolicy,
    GreedyInterceptOpponent,
    RandomDiscretePolicy,
    RandomValidOpponent,
    SafeDiscretePolicy,
    SafeHeuristicOpponent,
    make_baseline_policy,
    make_opponent,
)
from badminton1d.playback import FrameSnapshot, MatchTrace, RallyTrace, StageTrace, build_match_trace, build_rally_trace, interpolate_stage
from badminton1d.reset_sampling import ResetSamplingConfig
from badminton1d.reward_shaping import LoopPenaltyConfig, PressureRewardConfig
from badminton1d.render import ScoreboardOverlay
from badminton1d.selfplay import CheckpointPool, FixedCheckpointOpponent, FrozenCheckpointOpponent, LiveModelOpponent, MixedCheckpointOpponent
from badminton1d.rl_env import BadmintonRLEnv, RLEnvConfig, RewardConfig
from badminton1d.state import ShotAction, StageRecord, StageState
from badminton1d.video import (
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
    "MatchScore",
    "MatchTrace",
    "ObservationConfig",
    "ObservationEncoder",
    "CheckpointPool",
    "LoopPenaltyConfig",
    "MixedCheckpointOpponent",
    "RandomValidHitter",
    "RandomDiscretePolicy",
    "RandomValidOpponent",
    "ResetSamplingConfig",
    "PressureRewardConfig",
    "RewardConfig",
    "RallyResult",
    "RallyTrace",
    "RLEnvConfig",
    "SafeDiscretePolicy",
    "SafeHitter",
    "SafeHeuristicOpponent",
    "ShotAction",
    "SimulationConfig",
    "ScoreboardOverlay",
    "StageAgent",
    "StageRecord",
    "StageState",
    "StageTrace",
    "TrainingCurriculumSpec",
    "TrainingProgressSample",
    "VideoExportResult",
    "available_training_curricula",
    "build_match_trace",
    "build_training_curriculum",
    "build_rally_trace",
    "default_initial_state",
    "export_match_video",
    "export_rally_video",
    "export_training_progress_video",
    "interpolate_stage",
    "make_baseline_policy",
    "make_opponent",
    "render_match_frame",
    "render_video_frame",
    "reset_for_serve",
    "run_match",
    "run_rally",
]
