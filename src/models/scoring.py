"""
scoring.py — ContestScorer: stateless, atomic scoring engine.

Scoring rules (per CORE_PROBLEM_TO_SOLVE.md "Atomic Scoring Pipeline"):
  - Exact scoreline:            100 pts × stage_multiplier
  - Correct result (not exact): 50 pts × stage_multiplier
  - Wrong result:               0 pts

Stage multipliers:
  Group Stage          → 1.0×
  Round of 16          → 1.5×
  Quarter Final        → 2.0×
  Semi Final           → 3.0×
  Final / 3rd Place    → 4.0×
"""
import logging
from typing import Optional

from src.database import Prediction

logger = logging.getLogger(__name__)

# Stage multiplier lookup  — O(1) dict access
_STAGE_MULTIPLIERS: dict[str, float] = {
    "Group Stage": 1.0,
    "Round of 16": 1.5,
    "Quarter Final": 2.0,
    "Semi Final": 3.0,
    "Final / 3rd Place": 4.0,
}

_BASE_EXACT: int = 100
_BASE_RESULT: int = 50


def _result_sign(home: int, away: int) -> int:
    """Return +1 for home win, 0 for draw, -1 for away win."""
    if home > away:
        return 1
    elif home < away:
        return -1
    return 0


class ContestScorer:
    """Stateless scoring engine for the Contest Arena.

    Usage:
        scorer = ContestScorer()
        pts = scorer.calculate_score(prediction, actual_home=2, actual_away=1, stage="Group Stage")
    """

    def calculate_score(
        self,
        prediction: Prediction,
        actual_home: int,
        actual_away: int,
        stage: Optional[str] = None,
    ) -> int:
        """Calculate integer points for a single prediction vs. actual result.

        Args:
            prediction: The participant's locked Prediction.
            actual_home: Actual home goals (90-minute result; penalties not included).
            actual_away: Actual away goals.
            stage: Tournament stage name. Falls back to prediction.stage if None.

        Returns:
            Integer points earned (0, 50/75/…, or 100/150/…).
        """
        effective_stage = stage if stage is not None else prediction.stage
        multiplier = _STAGE_MULTIPLIERS.get(effective_stage, 1.0)

        p_home = prediction.predicted_home_score
        p_away = prediction.predicted_away_score

        # Exact scoreline match
        if p_home == actual_home and p_away == actual_away:
            score = int(_BASE_EXACT * multiplier)
            logger.debug(
                "Exact match for %s (match %d): %d pts.",
                prediction.participant_name,
                prediction.match_id,
                score,
            )
            return score

        # Correct result (win/draw/loss direction)
        if _result_sign(p_home, p_away) == _result_sign(actual_home, actual_away):
            score = int(_BASE_RESULT * multiplier)
            logger.debug(
                "Correct result for %s (match %d): %d pts.",
                prediction.participant_name,
                prediction.match_id,
                score,
            )
            return score

        # Wrong result
        logger.debug(
            "Wrong result for %s (match %d): 0 pts.",
            prediction.participant_name,
            prediction.match_id,
        )
        return 0
