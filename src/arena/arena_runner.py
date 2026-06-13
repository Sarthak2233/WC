"""
arena_runner.py — Atomic Scoring Orchestrator.

Implements the pipeline:
  1. HardLockEngine.ingest_result(match_id, home, away, stage)
  2. Load all Predictions for match_id
  3. Fan out ContestScorer.calculate_score() to every participant → O(N)
  4. Append scored rows to data/arena/scores_ledger.csv (immutable)
  5. Return updated LeaderboardSystem standings
"""
import csv
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.arena.hard_lock_engine import HardLockEngine, MatchResult
from src.database import Prediction, load_predictions
from src.models.leaderboard import LeaderboardSystem
from src.models.scoring import ContestScorer

logger = logging.getLogger(__name__)

SCORES_LEDGER_PATH = "data/arena/scores_ledger.csv"

_SCORE_FIELDS = [
    "participant_name",
    "match_id",
    "stage",
    "predicted_home",
    "predicted_away",
    "actual_home",
    "actual_away",
    "points",
    "is_exact",
    "is_knockout",
    "scored_at",
]

_KNOCKOUT_STAGES = {
    "Round of 16",
    "Quarter Final",
    "Semi Final",
    "Final / 3rd Place",
}


class ArenaRunner:
    """Atomic scoring orchestrator for the Contest Arena.

    Coordinates HardLockEngine, ContestScorer, and LeaderboardSystem in a
    single atomic pipeline per match result.

    Args:
        lock_engine: Optional HardLockEngine instance (created if not provided).
        scorer: Optional ContestScorer instance.
        leaderboard: Optional LeaderboardSystem instance.
        scores_path: Path to the scores ledger CSV.
        predictions_path: Path to the predictions ledger CSV.
    """

    def __init__(
        self,
        lock_engine: Optional[HardLockEngine] = None,
        scorer: Optional[ContestScorer] = None,
        leaderboard: Optional[LeaderboardSystem] = None,
        scores_path: str = SCORES_LEDGER_PATH,
        predictions_path: Optional[str] = None,
    ) -> None:
        self._lock_engine = lock_engine or HardLockEngine()
        self._scorer = scorer or ContestScorer()
        self._leaderboard = leaderboard or LeaderboardSystem()
        self._scores_path = scores_path
        self._predictions_path = predictions_path  # None → uses default in load_predictions

    def _append_score_row(self, row: Dict[str, Any]) -> None:
        """Append a single score event to the immutable scores ledger."""
        os.makedirs(os.path.dirname(self._scores_path), exist_ok=True)
        file_exists = os.path.isfile(self._scores_path)
        with open(self._scores_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_SCORE_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def process_result(
        self,
        match_id: int,
        home_goals: int,
        away_goals: int,
        stage: str,
    ) -> List[Dict[str, Any]]:
        """Ingest a result, score all predictions, and return updated standings.

        This is the primary entry point. The entire pipeline is atomic per
        match: ingest → fan-out score → persist → return leaderboard.

        Args:
            match_id: ID of the match that just ended.
            home_goals: Actual home goals.
            away_goals: Actual away goals.
            stage: Tournament stage name.

        Returns:
            Ranked list of participant records from LeaderboardSystem.
        """
        # Step 1: Ingest result into immutable ledger (raises on duplicate)
        result: MatchResult = self._lock_engine.ingest_result(
            match_id, home_goals, away_goals, stage
        )

        # Step 2: Load all predictions for this match — O(N)
        load_kwargs: Dict[str, Any] = {"match_id": match_id}
        if self._predictions_path is not None:
            load_kwargs["path"] = self._predictions_path
        predictions: List[Prediction] = load_predictions(**load_kwargs)

        if not predictions:
            logger.warning("No predictions found for match %d.", match_id)

        is_ko = stage in _KNOCKOUT_STAGES
        now_iso = datetime.now(timezone.utc).isoformat()
        scored_rows: List[Dict[str, Any]] = []

        # Step 3: Fan-out scoring — O(N) single pass
        for pred in predictions:
            pts = self._scorer.calculate_score(
                pred, actual_home=home_goals, actual_away=away_goals, stage=stage
            )
            is_exact = (
                pred.predicted_home_score == home_goals
                and pred.predicted_away_score == away_goals
            )

            score_row: Dict[str, Any] = {
                "participant_name": pred.participant_name,
                "match_id": match_id,
                "stage": stage,
                "predicted_home": pred.predicted_home_score,
                "predicted_away": pred.predicted_away_score,
                "actual_home": home_goals,
                "actual_away": away_goals,
                "points": pts,
                "is_exact": is_exact,
                "is_knockout": is_ko,
                "scored_at": now_iso,
            }

            # Step 4: Persist each score event atomically
            self._append_score_row(score_row)
            scored_rows.append(score_row)
            logger.info(
                "%s: %d pts for match %d (%d-%d actual).",
                pred.participant_name, pts, match_id, home_goals, away_goals,
            )

        # Step 5: Build and return updated leaderboard from scored_rows
        #         (enriched with first_pred_time from prediction)
        pred_time_map: Dict[str, datetime] = {
            p.participant_name: p.locked_at for p in predictions
        }
        enriched: List[Dict[str, Any]] = []
        for row in scored_rows:
            row_copy = dict(row)
            row_copy["first_pred_time"] = pred_time_map.get(
                row["participant_name"], datetime.now(timezone.utc)
            )
            enriched.append(row_copy)

        return self._leaderboard.build_from_scores_ledger(enriched)

    def get_current_leaderboard(self) -> List[Dict[str, Any]]:
        """Reconstruct the full leaderboard from the scores ledger (O(N)).

        Returns:
            Ranked participant records.
        """
        if not os.path.isfile(self._scores_path):
            return []

        scores: List[Dict[str, Any]] = []
        with open(self._scores_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                scores.append({
                    "participant_name": row["participant_name"],
                    "points": int(row["points"]),
                    "is_exact": row["is_exact"] == "True",
                    "is_knockout": row["is_knockout"] == "True",
                    "first_pred_time": datetime.fromisoformat(row["scored_at"]),
                })

        return self._leaderboard.build_from_scores_ledger(scores)
