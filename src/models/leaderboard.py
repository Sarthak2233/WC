"""
leaderboard.py — LeaderboardSystem: multi-key participant ranking.

Tie-breaking priority (per CORE_PROBLEM_TO_SOLVE.md):
  1. total_points DESC
  2. exact_hits DESC
  3. ko_points DESC  (points scored in knockout stages only)
  4. first_pred_time ASC  (earliest submission wins)
"""
import logging
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

ParticipantRecord = Dict[str, Any]


class LeaderboardSystem:
    """Ranks participants using a deterministic, multi-key sort in O(N log N).

    The sort key tuple ensures a stable, unambiguous ranking even with
    deeply nested ties.

    Usage:
        lb = LeaderboardSystem()
        ranked = lb.rank_participants(participants)
    """

    def rank_participants(
        self,
        participants: List[ParticipantRecord],
    ) -> List[ParticipantRecord]:
        """Return participants sorted by the tie-breaking priority chain.

        Args:
            participants: List of dicts, each containing:
                - name (str)
                - total_points (int)
                - exact_hits (int)
                - ko_points (int)
                - first_pred_time (datetime)

        Returns:
            New list sorted in descending rank order (index 0 = first place).
        """
        def _sort_key(p: ParticipantRecord):
            # Negative values for DESC sorts, positive for ASC
            return (
                -p["total_points"],
                -p["exact_hits"],
                -p["ko_points"],
                p["first_pred_time"],  # earlier datetime < later datetime → ASC natural
            )

        ranked = sorted(participants, key=_sort_key)
        logger.debug("Leaderboard computed for %d participants.", len(ranked))
        return ranked

    def build_from_scores_ledger(
        self,
        scores: List[Dict[str, Any]],
    ) -> List[ParticipantRecord]:
        """Aggregate per-match scores into participant records (O(N)).

        Args:
            scores: List of dicts with keys:
                - participant_name (str)
                - points (int)
                - is_exact (bool)
                - is_knockout (bool)
                - first_pred_time (datetime)

        Returns:
            Ranked list of ParticipantRecord dicts.
        """
        # O(N) aggregation via a plain dict
        aggregated: Dict[str, ParticipantRecord] = {}

        for row in scores:
            name: str = row["participant_name"]
            pts: int = row["points"]
            is_exact: bool = bool(row.get("is_exact", False))
            is_ko: bool = bool(row.get("is_knockout", False))
            pred_time: datetime = row["first_pred_time"]

            if name not in aggregated:
                aggregated[name] = {
                    "name": name,
                    "total_points": 0,
                    "exact_hits": 0,
                    "ko_points": 0,
                    "first_pred_time": pred_time,
                }

            rec = aggregated[name]
            rec["total_points"] += pts
            if is_exact:
                rec["exact_hits"] += 1
            if is_ko:
                rec["ko_points"] += pts
            # Keep the earliest submission time
            if pred_time < rec["first_pred_time"]:
                rec["first_pred_time"] = pred_time

        return self.rank_participants(list(aggregated.values()))
