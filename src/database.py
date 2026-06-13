"""
database.py — Prediction dataclass with CSV-backed, append-only persistence.

Per CORE_PROBLEM_TO_SOLVE.md: all predictions are immutable events in an
append-only ledger. No row is ever overwritten or deleted.
"""
import csv
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)

PREDICTIONS_PATH = "data/arena/predictions.csv"

_FIELDNAMES = [
    "participant_name",
    "match_id",
    "predicted_home_score",
    "predicted_away_score",
    "locked_at",
    "stage",
]


@dataclass
class Prediction:
    """Represents a single participant prediction for a match.

    Attributes:
        participant_name: Unique identifier for the participant (or "ORACLE").
        match_id: Integer ID of the match being predicted.
        predicted_home_score: Predicted integer home goals.
        predicted_away_score: Predicted integer away goals.
        locked_at: UTC timestamp when prediction was submitted (auto-set).
        stage: Tournament stage name (e.g., "Group Stage", "Final / 3rd Place").
    """

    participant_name: str
    match_id: int
    predicted_home_score: int
    predicted_away_score: int
    locked_at: datetime = field(default_factory=datetime.utcnow)
    stage: str = "Group Stage"

    def to_dict(self) -> dict:
        """Serialise prediction to a plain dict for CSV export."""
        d = asdict(self)
        d["locked_at"] = self.locked_at.isoformat()
        return d


def append_prediction(prediction: Prediction, path: str = PREDICTIONS_PATH) -> None:
    """Append a single Prediction to the immutable CSV ledger (O(1)).

    Args:
        prediction: The Prediction instance to persist.
        path: Override for the default CSV path (useful in tests).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(prediction.to_dict())

    logger.debug("Appended prediction for %s / match %d.", prediction.participant_name, prediction.match_id)


def load_predictions(
    match_id: Optional[int] = None,
    path: str = PREDICTIONS_PATH,
) -> List[Prediction]:
    """Load predictions from the CSV ledger in O(N).

    Args:
        match_id: If provided, filters to only predictions for that match.
        path: Override for the default CSV path.

    Returns:
        List of Prediction instances.
    """
    if not os.path.isfile(path):
        return []

    results: List[Prediction] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = int(row["match_id"])
            if match_id is not None and mid != match_id:
                continue
            results.append(
                Prediction(
                    participant_name=row["participant_name"],
                    match_id=mid,
                    predicted_home_score=int(row["predicted_home_score"]),
                    predicted_away_score=int(row["predicted_away_score"]),
                    locked_at=datetime.fromisoformat(row["locked_at"]),
                    stage=row["stage"],
                )
            )
    return results
